"""Turn predictions into the published rankings: market blend, reasons, JSON."""
import datetime as dt
import json

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, SCORING
from .live import _norm, market_means
from .features import score
from .model import COMPONENTS, QUANTILES

MARKET_WEIGHT = 0.5  # share of the market-vs-model gap applied when a prop exists (a judgment call)
VEGAS_LABEL = "Betting market"

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
    "qb": "This week's starting QB's longer-run efficiency (EPA per dropback, weighted toward "
          "recent games) compared with the QB play behind this player's recent stats. With a new "
          "starter, this is the upgrade or downgrade. With the same starter, it is the model "
          "expecting his recent form to drift back toward his longer track record. "
          "Baseline: the same quality of QB play as recent weeks.",
    "weather": "Kickoff forecast for outdoor stadiums (wind and temperature). Domes and retractable "
               "roofs are neutral. Baseline: 65°F with 5 mph wind.",
    "health": "The player's own injury designation and practice participation this week. "
              "Baseline: healthy and practicing fully.",
}


VEGAS_KEY = ("Sportsbooks post lines for some players' receptions, yards and anytime TD. We turn each line "
             "into an expected stat and compare it with our model's own estimate of that stat. Where they "
             "disagree, we split the difference: half the gap is added to the projection. Where they agree, "
             "nothing changes. Unlike the factors above, this is the actual change made to the model's "
             "number, not a sensitivity.")


def explain(model, live, train):
    """Points each feature group adds vs. a neutral baseline, per scoring format.

    These are sensitivities: re-predict with one group reset to its baseline,
    everything else held fixed. They are not causal and don't sum to the
    projection (the model has interactions). Returns {fmt: DataFrame}.
    """
    base_pred = model.predict(live)
    matchup_cols = [c for c in model.features if c.startswith("def_")]
    recent = train[train.season >= train.season.max() - 1]
    out = {fmt: {} for fmt in SCORING}
    no_forecast = (live.dome == 0) & live.temp.isna()  # unknown weather: no effect to explain
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
        alt_pred = model.predict(alt)
        for fmt in SCORING:
            eff = base_pred[f"{fmt}_proj"].to_numpy() - alt_pred[f"{fmt}_proj"].to_numpy()
            if key == "weather":
                eff = np.where(no_forecast, 0.0, eff)
            out[fmt][key] = eff
    return {fmt: pd.DataFrame(v, index=live.index) for fmt, v in out.items()}


def blend_market(pred, live_players, raw_props):
    """Adjust projections toward Vegas player props where available.

    The market view is the model's own component stat line with each
    prop-implied stat substituted in. The adjustment is MARKET_WEIGHT times
    (market view − the model's component view), added on top of the full
    projection. So a prop that matches the model changes nothing; only the
    disagreement moves the number. Model-only numbers stay in model_{fmt}.
    """
    mk = market_means(raw_props)
    pred["market"] = False
    for fmt in SCORING:
        pred[f"model_{fmt}"] = pred[f"{fmt}_proj"]
        pred[f"{fmt}_vegas"] = 0.0
    for c in COMPONENTS:
        pred[f"model_{c}"] = pred[f"proj_{c}"]
    pred["model_tds"] = pred.proj_rec_tds + pred.proj_rush_tds
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
    model_comp = pd.DataFrame({c: pred[f"proj_{c}"] for c in COMPONENTS})
    for fmt in SCORING:
        adj = (MARKET_WEIGHT * (score(mcomp, fmt) - score(model_comp, fmt))).where(has, 0.0)
        pred[f"{fmt}_vegas"] = adj
        pred[f"{fmt}_proj"] = (pred[f"{fmt}_proj"] + adj).clip(lower=0)
    for c in COMPONENTS:  # displayed stat line
        pred.loc[has, f"proj_{c}"] = (model_comp[c] + MARKET_WEIGHT * (mcomp[c] - model_comp[c]))[has]
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
            "qb": _qb_text(r, names),
            "weather": "Dome / roof" if dome else ("Forecast unavailable" if pd.isna(r.temp) or pd.isna(r.wind)
                                                  else f"{r.temp:.0f}°F, wind {r.wind:.0f} mph"),
            "health": {0: "No injury designation", 1: "Questionable", 2: "Doubtful"}.get(int(r.inj_report), "Listed")
                      + {0: "", 1: ", limited practice", 2: ", did not practice"}.get(int(r.inj_practice), ""),
        }
        out[i] = {GROUPS[k][1]: v for k, v in ctx.items()}
    return out


def _qb_text(r, names):
    qb = names.get(r.qb_id, "Unknown QB")
    if pd.isna(r.qb_epa) or pd.isna(r.tm_epa_per_db):
        return f"{qb}: not enough history to compare"
    if r.qb_change:
        return (f"New starter {qb}: {r.qb_epa:+.2f} EPA/dropback over his longer run vs. "
                f"{r.team}'s recent {r.tm_epa_per_db:+.2f}")
    if abs(r.qb_epa - r.tm_epa_per_db) < 0.03:
        return f"Same starter, {qb}, playing at his usual level ({r.qb_epa:+.2f} EPA/dropback)"
    trend = "above" if r.tm_epa_per_db > r.qb_epa else "below"
    drift = "some regression" if trend == "above" else "a bounce-back"
    return (f"Same starter, {qb}, playing {trend} his usual level lately ({r.tm_epa_per_db:+.2f} vs. "
            f"his longer-run {r.qb_epa:+.2f} EPA/dropback); the model expects {drift}")


def _vegas_text(r):
    if not r.get("market", False):
        return "No betting lines for this player; projection is our model alone"
    bits = []
    for k, lab, f in (("receptions", "rec", "{:.1f}"), ("rec_yards", "rec yds", "{:.0f}"),
                      ("rush_yards", "rush yds", "{:.0f}")):
        m = r.get(f"mkt_{k}")
        if m is not None and pd.notna(m):
            bits.append(f"{lab} {f.format(m)} (model {f.format(r[f'model_{k}'])})")
    t = r.get("mkt_tds")
    if t is not None and pd.notna(t):
        bits.append(f"TD {1 - np.exp(-t):.0%} (model {1 - np.exp(-r['model_tds']):.0%})")
    return f"Betting lines vs. our model: {'; '.join(bits)}. We move halfway toward the lines"


def to_json(pred, reasons, live_players, games, season, week, meta, ctx=None):
    """reasons: {fmt: DataFrame of factor effects} from explain()."""
    info = live_players.set_index("player_id")
    gm = {}
    for g in games.itertuples():
        for team, opp, home in ((g.home_team, g.away_team, True), (g.away_team, g.home_team, False)):
            gm[team] = {"opp": opp, "home": home, "kickoff": f"{g.gameday} {g.gametime}"}
    players = []
    for i, r in pred.iterrows():
        pid = r.player_id
        p = {
            "id": pid, "name": info.name.get(pid, pid), "pos": r.position, "team": r.team,
            "opp": r.opp, "home": gm.get(r.team, {}).get("home"),
            "kickoff": gm.get(r.team, {}).get("kickoff"),
            "headshot": info.headshot.get(pid),
            "questionable": bool(info.inj_report.get(pid, 0)),
            "market": bool(r.get("market", False)),
            "stats": {k: round(float(r[f"proj_{k}"]), 2) for k in
                      ("receptions", "rec_yards", "rec_tds", "carries", "rush_yards", "rush_tds")},
            "context": {**(ctx or {}).get(i, {}), VEGAS_LABEL: _vegas_text(r)},
            "vegas": {k: round(float(r[f"mkt_{k}"]), 2) for k in ("receptions", "rec_yards", "rush_yards", "tds")
                      if f"mkt_{k}" in r and pd.notna(r[f"mkt_{k}"])},
        }
        for fmt in SCORING:
            rs = reasons[fmt].loc[i].rename(lambda k: GROUPS[k][1])
            vegas = float(r.get(f"{fmt}_vegas", 0.0))
            chips = pd.concat([rs, pd.Series({VEGAS_LABEL: vegas})])
            top = chips[chips.abs() >= 0.3]
            top = top.reindex(top.abs().sort_values(ascending=False).index).head(3)
            p[fmt] = {
                "proj": round(float(r[f"{fmt}_proj"]), 1),
                "model": round(float(r.get(f"model_{fmt}", r[f"{fmt}_proj"])), 1),
                "vegas": round(vegas, 2),
                "q": [round(float(r[f"{fmt}_q{int(q * 100)}"]), 1) for q in QUANTILES],
                "boom": round(float(r[f"{fmt}_boom"]), 3),
                "reasons": [{"label": k, "pts": round(float(v), 1)} for k, v in top.items()],
                "factors": {k: round(float(v), 2) for k, v in rs.items()},
            }
        players.append(p)
    return {
        "season": season, "week": week,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "quantiles": QUANTILES, "boom_threshold": 20,
        "factor_key": [{"label": GROUPS[k][1], "text": t,
                        "typical": round(float(reasons["half"].loc[pred.half_proj >= 8, k].abs().median()), 2),
                        "max": round(float(reasons["half"].loc[pred.half_proj >= 8, k].abs().max()), 1)}
                       for k, t in FACTOR_KEY.items()]
                      + [{"label": VEGAS_LABEL, "text": VEGAS_KEY,
                          "typical": round(float(pred.loc[pred.market & (pred.half_proj >= 8), "half_vegas"].abs().median()), 2)
                          if (pred.market & (pred.half_proj >= 8)).any() else 0.0,
                          "max": round(float(pred.loc[pred.half_proj >= 8, "half_vegas"].abs().max()), 1)
                          if (pred.half_proj >= 8).any() else 0.0}],
        "vegas_label": VEGAS_LABEL, **meta, "players": players,
    }


def write(payload):
    """The current rankings for the page. Per-run history is archive.py's job (immutable)."""
    data_dir = OUTPUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), default=lambda o: None if pd.isna(o) else str(o))
    (data_dir / "rankings.json").write_text(text)
