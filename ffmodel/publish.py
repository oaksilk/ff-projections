"""Turn predictions into the published rankings: market blend, reasons, JSON."""
import datetime as dt
import json

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, SCORING
from .live import _norm, market_means
from .features import score
from .model import COMPONENTS, QUANTILES

MARKET_WEIGHT = 0.5  # share of each component taken from Vegas when a prop exists

# Feature groups we "neutralize" one at a time to explain a projection.
GROUPS = {
    "game": (["spread", "total_line", "implied", "opp_implied"], "Game environment"),
    "matchup": (None, "Opponent defense"),
    "secondary": (["db_out_share", "db_out_n"], "Opponent DB injuries"),
    "teammates": (["vac_tgt", "vac_car"], "Teammate absences"),
    "qb": (["qb_change", "qb_epa", "qb_games"], "Starting QB"),
    "weather": (["temp", "wind"], "Weather"),
    "health": (["inj_report", "inj_practice"], "Own injury status"),
}


def explain(model, live, train):
    """Half-PPR points each feature group adds vs. a neutral baseline."""
    base = model.predict(live)["half_proj"].to_numpy()
    matchup_cols = [c for c in model.features if c.startswith("def_")]
    recent = train[train.season >= train.season.max() - 1]
    out = {}
    for key, (cols, _) in GROUPS.items():
        cols = matchup_cols if cols is None else cols
        alt = live.copy()
        for c in cols:
            if key in ("secondary", "teammates", "health"):
                alt[c] = 0
            elif key == "qb":  # an established, league-median starter
                alt[c] = {"qb_change": 0, "qb_games": 40}.get(c, recent.qb_epa.median())
            elif key == "weather":
                alt[c] = {"temp": 65.0, "wind": 5.0}[c]
            else:  # league-typical value for this position
                med = recent.groupby("position")[c].median()
                alt[c] = alt.position.map(med)
        out[key] = base - model.predict(alt)["half_proj"].to_numpy()
    return pd.DataFrame(out, index=live.index)


def blend_market(pred, live_players, raw_props):
    """Blend projections with Vegas player props where available.

    The market view is the model's stat line with each prop-implied stat
    substituted in; the published projection is MARKET_WEIGHT of that.
    """
    mk = market_means(raw_props)
    pred["market"] = False
    if mk.empty:
        return pred
    mk = mk.drop_duplicates("name_key").set_index("name_key")
    keys = pred.player_id.map(live_players.set_index("player_id").name).map(_norm)
    mcomp = pd.DataFrame({c: pred[f"proj_{c}"] for c in COMPONENTS})
    for stat in ("receptions", "rec_yards", "rush_yards"):
        if stat in mk:
            m = keys.map(mk[stat])
            pred[f"mkt_{stat}"] = m
            mcomp[stat] = m.fillna(mcomp[stat])
            pred["market"] |= m.notna()
    if "tds" in mk:
        m = keys.map(mk["tds"])
        pred["mkt_tds"] = m
        model_tds = (mcomp.rec_tds + mcomp.rush_tds).clip(lower=1e-6)
        scale = (m / model_tds).fillna(1)
        mcomp["rec_tds"] *= scale
        mcomp["rush_tds"] *= scale
        pred["market"] |= m.notna()
    has = pred.market
    for fmt in SCORING:
        pred.loc[has, f"{fmt}_proj"] = ((1 - MARKET_WEIGHT) * pred[f"{fmt}_proj"]
                                        + MARKET_WEIGHT * score(mcomp, fmt))[has]
    for c in COMPONENTS:  # displayed stat line
        pred.loc[has, f"proj_{c}"] = ((1 - MARKET_WEIGHT) * pred[f"proj_{c}"] + MARKET_WEIGHT * mcomp[c])[has]
    return pred


def to_json(pred, reasons, live_players, games, season, week, meta):
    info = live_players.set_index("player_id")
    gm = {}
    for g in games.itertuples():
        for team, opp, home in ((g.home_team, g.away_team, True), (g.away_team, g.home_team, False)):
            gm[team] = {"opp": opp, "home": home, "kickoff": f"{g.gameday} {g.gametime}"}
    players = []
    for i, r in pred.iterrows():
        pid = r.player_id
        rs = reasons.loc[i]
        top = rs[rs.abs() >= 0.3]
        top = top.reindex(top.abs().sort_values(ascending=False).index).head(3)
        p = {
            "id": pid, "name": info.name.get(pid, pid), "pos": r.position, "team": r.team,
            "opp": r.opp, "home": gm.get(r.team, {}).get("home"),
            "kickoff": gm.get(r.team, {}).get("kickoff"),
            "headshot": info.headshot.get(pid),
            "questionable": bool(info.inj_report.get(pid, 0)),
            "market": bool(r.get("market", False)),
            "stats": {k: round(float(r[f"proj_{k}"]), 2) for k in
                      ("receptions", "rec_yards", "rec_tds", "carries", "rush_yards", "rush_tds")},
            "reasons": [{"label": GROUPS[k][1], "pts": round(float(v), 1)} for k, v in top.items()],
            "factors": {GROUPS[k][1]: round(float(v), 2) for k, v in rs.items()},
            "model_half": round(float(r.get("model_half", r["half_proj"])), 1),
            "vegas": {k: round(float(r[f"mkt_{k}"]), 2) for k in ("receptions", "rec_yards", "rush_yards", "tds")
                      if f"mkt_{k}" in r and pd.notna(r[f"mkt_{k}"])},
        }
        for fmt in SCORING:
            p[fmt] = {
                "proj": round(float(r[f"{fmt}_proj"]), 1),
                "q": [round(float(r[f"{fmt}_q{int(q * 100)}"]), 1) for q in QUANTILES],
                "boom": round(float(r[f"{fmt}_boom"]), 3),
            }
        players.append(p)
    return {
        "season": season, "week": week,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "quantiles": QUANTILES, "boom_threshold": 20, **meta, "players": players,
    }


def write(payload, season, week):
    data_dir = OUTPUT_DIR / "data"
    (data_dir / "history").mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), default=lambda o: None if pd.isna(o) else str(o))
    (data_dir / "rankings.json").write_text(text)
    (data_dir / "history" / f"{season}_w{week:02d}.json").write_text(text)
