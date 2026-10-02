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
    "qb": (["qb_change", "qb_epa", "qb_upgrade"], "QB vs. recent QB play"),
    "weather": (["temp", "wind"], "Weather"),
    "health": (["inj_report", "inj_practice"], "Own injury status"),
}


# Plain-English definitions shown in the page's factor key.
FACTOR_KEY = {
    "game": "Vegas spread and over/under, turned into this team's implied points. Higher-scoring "
            "expected games mean more plays and more scoring chances. Baseline: a typical game "
            "for the position over the last two seasons.",
    "matchup": "How this opponent's defense has played lately: fantasy points and targets allowed "
               "to this position, EPA per dropback allowed, yards per carry and per target allowed, "
               "and how many plays opponents run against them. Baseline: a league-typical defense.",
    "secondary": "Regular defensive backs (50%+ of snaps recently) on the opposing team who are "
                 "ruled out or on IR. Baseline: a fully healthy secondary.",
    "teammates": "Share of the team's recent targets and carries belonging to teammates who are out "
                 "this week, which frees up opportunities. Baseline: nobody missing.",
    "qb": "This week's starting QB's efficiency (EPA per dropback over his recent career) compared "
          "with the QB play behind this player's recent stats. Positive means an upgrade. "
          "Baseline: the same quality of QB play as recent weeks.",
    "weather": "Kickoff forecast for outdoor stadiums (wind and temperature). Domes and retractable "
               "roofs are neutral. Baseline: 65°F with 5 mph wind.",
    "health": "The player's own injury designation and practice participation this week. "
              "Baseline: healthy and practicing fully.",
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
            elif key == "qb":  # same quality of QB play the offense has had recently
                alt[c] = {"qb_change": 0, "qb_upgrade": 0, "qb_epa": alt.tm_epa_per_db}[c]
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


def context(live, train, names):
    """Short human-readable inputs behind each factor, per player row."""
    recent = train[train.season >= train.season.max() - 1]
    typ_implied = recent.implied.median()
    # Rank this week's opponents by fantasy points allowed to each position (1 = most generous).
    opps = live.drop_duplicates(["position", "opp"])[["position", "opp", "def_alw_fp_pos"]].copy()
    opps["rank"] = opps.groupby("position").def_alw_fp_pos.rank(ascending=False, method="min")
    opps["n"] = opps.groupby("position").opp.transform("size")
    opp_rank = opps.set_index(["position", "opp"])
    out = {}
    for i, r in live.iterrows():
        dome = r.dome == 1
        ctx = {
            "game": f"{r.team} implied {r.implied:.1f} pts (typical {typ_implied:.1f}); "
                    f"spread {'+' if r.spread < 0 else '−' if r.spread > 0 else ''}{abs(r.spread):g}, total {r.total_line:g}",
            "matchup": f"{r.opp} allows {r.def_alw_fp_pos:.1f} half-PPR pts/game to {r.position}s lately "
                       f"(#{int(opp_rank.loc[(r.position, r.opp), 'rank'])} most of "
                       f"{int(opp_rank.loc[(r.position, r.opp), 'n'])} opponents this week)",
            "secondary": f"{int(r.db_out_n)} regular {r.opp} DB{'s' if r.db_out_n != 1 else ''} out",
            "teammates": f"{r.vac_tgt:.0%} of targets and {r.vac_car:.0%} of carries vacated by injured teammates",
            "qb": f"{names.get(r.qb_id, 'Unknown QB')} {r.qb_epa:+.2f} EPA/dropback vs. "
                  f"{r.team}'s recent {r.tm_epa_per_db:+.2f}",
            "weather": "Dome / roof" if dome else f"{r.temp:.0f}°F, wind {r.wind:.0f} mph",
            "health": {0: "No injury designation", 1: "Questionable", 2: "Doubtful"}.get(int(r.inj_report), "Listed")
                      + {0: "", 1: ", limited practice", 2: ", did not practice"}.get(int(r.inj_practice), ""),
        }
        out[i] = {GROUPS[k][1]: v for k, v in ctx.items()}
    return out


def to_json(pred, reasons, live_players, games, season, week, meta, ctx=None):
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
            "context": (ctx or {}).get(i, {}),
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
        "quantiles": QUANTILES, "boom_threshold": 20,
        "factor_key": [{"label": GROUPS[k][1], "text": t,
                        "typical": round(float(reasons.loc[pred.half_proj >= 8, k].abs().median()), 2),
                        "max": round(float(reasons.loc[pred.half_proj >= 8, k].abs().max()), 1)}
                       for k, t in FACTOR_KEY.items()],
        **meta, "players": players,
    }


def write(payload, season, week):
    data_dir = OUTPUT_DIR / "data"
    (data_dir / "history").mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), default=lambda o: None if pd.isna(o) else str(o))
    (data_dir / "rankings.json").write_text(text)
    (data_dir / "history" / f"{season}_w{week:02d}.json").write_text(text)
