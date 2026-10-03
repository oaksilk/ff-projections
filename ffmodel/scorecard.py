"""Prospective scorecard: grade every archived forecast source on finished weeks.

Sources: our model alone, our published numbers (model + Vegas adjustment),
FantasyPros ECR, the recent-average baseline, the model + ECR rank blend,
Sleeper (Rotowire) and ESPN.

Rules (decided before any live week was graded):
- Each player is graded on each source's last archive file saved *before his
  own kickoff* (archive.py). Nothing after kickoff counts.
- The decision universe at a position is every player in *any* source's top N
  (48 WR / 36 RB / 18 TE). A player recommended who didn't play scores 0, so
  sources are penalized for recommending inactive players.
- A player a source didn't rank sits below everything that source ranked.
- Pairwise accuracy gives tied forecasts half credit (metrics.py). Flex is
  cross-position pairs only; ECR and the blend sit out of flex because the live
  ECR feed has no cross-position ranking.
- ECR is a PPR ranking; it is graded against all three formats as-is.

Output: site/data/scorecard.json (accuracy figures only, no third-party
numbers), shown on the page and in the weekly email.
"""
import datetime as dt
import json
import logging
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .archive import FORECAST_DIR, THIRD_PARTY_DIR
from .backtest import TOP_N
from .config import OUTPUT_DIR, SCORING
from .metrics import bootstrap_diff, decision_metrics, h2h_prob, pinball, rank_blend
from .model import QUANTILES

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
OUT = OUTPUT_DIR / "data" / "scorecard.json"
SOURCES = {
    "published": "Our rankings (model + Vegas)",
    "model": "Our model alone",
    "blend": "Model + expert blend",
    "ecr": "FantasyPros experts",
    "sleeper": "Sleeper (Rotowire)",
    "espn": "ESPN",
    "naive": "Recent average",
}
POINT_SOURCES = ("published", "model", "sleeper", "espn", "naive")
UNRANKED = -1e6


def _ts(path):
    return pd.Timestamp(dt.datetime.strptime(path.stem, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=dt.timezone.utc))


def _files(base, season, week):
    d = base / f"{season}_w{week:02d}"
    return sorted(d.glob("*.json"), key=_ts) if d.is_dir() else []


def archived_weeks():
    out = []
    for d in sorted(FORECAST_DIR.glob("*_w*")):
        if d.is_dir() and any(d.glob("*.json")):
            s, w = d.name.split("_w")
            out.append((int(s), int(w)))
    return out


def kickoffs(d, season, week):
    """team -> kickoff time (UTC) for the week."""
    s = d["schedules"]
    g = s[(s.season == season) & (s.week == week) & (s.game_type == "REG")]
    out = {}
    for r in g.itertuples():
        t = pd.Timestamp(f"{r.gameday} {r.gametime}").tz_localize(ET).tz_convert("UTC")
        out[r.home_team] = out[r.away_team] = t
    return out


def _latest_before(files, kick):
    """The last file saved strictly before `kick`, or None."""
    best = None
    for f in files:
        if _ts(f) < kick:
            best = f
    return best


def build_week(d, season, week, actual):
    """One row per player in any source: team, position, each source's numbers, actual points."""
    kick = kickoffs(d, season, week)
    ffiles, tfiles = _files(FORECAST_DIR, season, week), _files(THIRD_PARTY_DIR, season, week)
    cache = {}

    def load(f):
        if f not in cache:
            cache[f] = json.loads(f.read_text())
        return cache[f]

    # Team for every candidate: our forecasts, then third parties, then the weekly roster.
    roster = d["rosters"]
    roster = roster[(roster.season == season) & (roster.week == week)].drop_duplicates("gsis_id")
    team_of = dict(zip(roster.gsis_id, roster.team))
    rows = {}

    def row(pid, pos=None, team=None):
        r = rows.setdefault(pid, {"player_id": pid})
        r.setdefault("pos", pos)
        r.setdefault("team", team or team_of.get(pid))
        return r

    for f in ffiles:
        for p in load(f)["players"]:
            row(p["id"], p["pos"], p["team"])
    for f in tfiles:
        rec = load(f)
        for src, pos_key in (("ecr", "pos"), ("sleeper", "pos"), ("espn", "pos")):
            for x in rec.get(src) or []:
                if x.get("gsis_id") and x.get(pos_key) in TOP_N:
                    row(x["gsis_id"], x[pos_key], x.get("team"))

    for pid, r in rows.items():
        k = kick.get(r["team"])
        if k is None:  # bye, or team unknown: can't decide what was "before kickoff"
            continue
        r["kickoff"] = k
        f = _latest_before(ffiles, k)
        if f is not None:
            p = next((p for p in load(f)["players"] if p["id"] == pid), None)
            if p:
                for fmt in SCORING:
                    r[f"model_{fmt}"] = p["model"][fmt]["proj"]
                    r[f"published_{fmt}"] = p["published"][fmt]["proj"]
                    r[f"naive_{fmt}"] = p["naive"][fmt]
                    for kind in ("model", "published"):
                        r[f"{kind}_{fmt}_q"] = p[kind][fmt]["q"]
                        r[f"{kind}_{fmt}_boom"] = p[kind][fmt]["boom"]
        t = _latest_before(tfiles, k)
        if t is not None:
            rec = load(t)
            for x in rec.get("ecr") or []:
                if x.get("gsis_id") == pid:
                    r["ecr"] = x["ecr"]
            for src in ("sleeper", "espn"):
                for x in rec.get(src) or []:
                    if x.get("gsis_id") == pid:
                        for fmt in SCORING:
                            r[f"{src}_{fmt}"] = x.get(fmt)
    df = pd.DataFrame([r for r in rows.values() if "kickoff" in r and r.get("pos") in TOP_N])
    if df.empty:
        return df
    df = df.merge(actual, on="player_id", how="left")
    df["played"] = df.ppr.notna()
    for fmt in SCORING:
        df[f"actual_{fmt}"] = df[fmt].fillna(0.0)  # recommended but didn't play = 0
    return df


def _score(g, src, fmt):
    if src == "ecr":
        if "ecr" not in g or g.ecr.isna().all():
            return None
        s = -g.ecr
    elif src == "blend":
        if "ecr" not in g or g.ecr.isna().all():
            return None
        m, e = g.get(f"model_{fmt}"), -g.ecr
        ok = m.notna() & e.notna()
        s = pd.Series(np.nan, index=g.index)
        s[ok] = rank_blend(m[ok].to_numpy(), e[ok].to_numpy())
    else:
        col = f"{src}_{fmt}"
        if col not in g or g[col].isna().all():
            return None
        s = g[col]
    return s.astype(float).fillna(UNRANKED)


def _universe(df, fmt):
    """Union of every source's top N at each position."""
    keep = set()
    for pos, n in TOP_N.items():
        g = df[df.pos == pos]
        for src in SOURCES:
            s = _score(g, src, fmt)
            if s is not None:
                keep |= set(s[s > UNRANKED].nlargest(n).index)
    return df.loc[sorted(keep)]


def grade_week(df):
    """Rows of (group, fmt, source, metrics) for one week."""
    rows = []
    for fmt in SCORING:
        u = _universe(df, fmt)
        groups = [(pos, u[u.pos == pos], False) for pos in TOP_N] + [("FLEX", u, True)]
        for grp, g, flex in groups:
            if len(g) < 8:
                continue
            scores = {}
            for src in SOURCES:
                if flex and src in ("ecr", "blend"):
                    continue
                s = _score(g, src, fmt)
                if s is not None:
                    scores[src] = s.to_numpy()
            ecr_rank = g.ecr.rank(method="first").to_numpy() if "ecr" in scores and not flex else None
            m = decision_metrics(scores, g[f"actual_{fmt}"].to_numpy(), ecr_rank,
                                 cross=g.pos.to_numpy() if flex else None)
            for src in scores:
                r = {"group": grp, "fmt": fmt, "source": src, "n": len(g),
                     "pair": m.get(f"{src}_pair"), "lost": m.get(f"{src}_lost"),
                     "close": m.get(f"{src}_close"), "close_lost": m.get(f"{src}_close_lost")}
                if src in POINT_SOURCES and f"{src}_{fmt}" in g:
                    have = g[f"{src}_{fmt}"].notna()
                    r["mae"] = float(np.mean(np.abs(g.loc[have, f"{src}_{fmt}"] - g.loc[have, f"actual_{fmt}"]))) \
                        if have.any() else None
                n_top = TOP_N.get(grp)
                if n_top:
                    top = pd.Series(scores[src], index=g.index).nlargest(n_top).index
                    r["recommended_dnp"] = int((~g.loc[top, "played"]).sum())
                rows.append(r)
    return rows


def grade_distributions(df, slope):
    """Range, boom and head-to-head checks for our two forecasts (players we projected)."""
    out = []
    for kind in ("model", "published"):
        for fmt in SCORING:
            col = f"{kind}_{fmt}_q"
            if col not in df:
                continue
            g = df[df[col].notna()]
            g = g[g[f"{kind}_{fmt}"] >= 5]  # meaningful players; fringe ones swamp the averages
            if len(g) < 20:
                continue
            q = np.array(g[col].tolist(), float)
            a = g[f"actual_{fmt}"].to_numpy()
            r = {"source": kind, "fmt": fmt, "n": len(g),
                 "coverage_10_90": float(np.mean((a >= q[:, 0]) & (a <= q[:, -1]))),
                 "pinball": float(np.mean([pinball(a, q[:, k], qq) for k, qq in enumerate(QUANTILES)])),
                 "boom_pred": float(g[f"{kind}_{fmt}_boom"].mean()), "boom_hit": float(np.mean(a > 20))}
            briers = []
            for pos in TOP_N:
                pg = g[g.pos == pos]
                i, j = np.triu_indices(len(pg), 1)
                aa = pg[f"actual_{fmt}"].to_numpy()
                keep = aa[i] != aa[j]
                if keep.sum() == 0:
                    continue
                qq = np.array(pg[col].tolist(), float)
                p = h2h_prob(qq[i[keep]], qq[j[keep]])
                z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1))
                p = 1 / (1 + np.exp(-slope * z))
                briers.append((p - (aa[i[keep]] > aa[j[keep]])) ** 2)
            if briers:
                r["h2h_brier"] = float(np.mean(np.concatenate(briers)))
            out.append(r)
    return out


def _summary(weeks):
    """Season-to-date averages per source (ALL = RB/WR/TE weeks equally weighted), with
    95% intervals vs. the experts from resampling weeks."""
    rows = pd.DataFrame([{**r, "season": w["season"], "week": w["week"]} for w in weeks for r in w["rows"]])
    if rows.empty:
        return {}
    out = {}
    for season, s in rows.groupby("season"):
        pos = s[s.group.isin(TOP_N)]
        res = {"weeks": int(s.week.nunique()), "sources": {}}
        for fmt in SCORING:
            f = pos[pos.fmt == fmt]
            wide = f.pivot_table(index=["season", "week", "group"], columns="source", values="pair").reset_index()
            for src in SOURCES:
                g = f[f.source == src]
                if g.empty:
                    continue
                e = {"pair": g.pair.mean(), "close": g.close.mean(), "lost": g.lost.mean(),
                     "mae": g.mae.mean() if "mae" in g else None,
                     "flex_pair": s[(s.group == "FLEX") & (s.fmt == fmt) & (s.source == src)].pair.mean(),
                     "recommended_dnp": int(g.recommended_dnp.sum()) if "recommended_dnp" in g else None}
                if src != "ecr" and "ecr" in wide and src in wide:
                    mean, lo, hi = bootstrap_diff(wide, src, "ecr")
                    e["vs_ecr"] = {"diff": mean, "lo": lo, "hi": hi}
                res["sources"].setdefault(src, {})[fmt] = {
                    k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v) for k, v in e.items()}
        out[str(season)] = res
    return out


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {"weeks": []}


def update(d, slope=None):
    """Grade every archived week that is final and not yet graded. Returns new (season, week)s."""
    from .evaluate import actuals, week_final
    if slope is None:
        from .config import ROOT
        slope = json.loads((ROOT / "model_data" / "h2h_calibration.json").read_text())["slope"]
    sc = load()
    done = {(w["season"], w["week"]) for w in sc["weeks"]}
    new = []
    for season, week in archived_weeks():
        if (season, week) in done or not week_final(d, season, week):
            continue
        df = build_week(d, season, week, actuals(d, season, week))
        if df.empty:
            continue
        sc["weeks"].append({"season": season, "week": week, "players": int(len(df)),
                            "played": int(df.played.sum()), "rows": grade_week(df),
                            "distributions": grade_distributions(df, slope)})
        new.append((season, week))
        log.info("scorecard: graded %s week %s (%d players)", season, week, len(df))
    if new or not OUT.exists():
        sc["weeks"].sort(key=lambda w: (w["season"], w["week"]))
        sc.update({"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
                   "source_names": SOURCES, "summary": _summary(sc["weeks"])})
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(_clean(sc), separators=(",", ":")))
    return new


def markdown(season, week):
    """Plain-English scorecard section for the weekly email."""
    sc = load()
    wk = next((w for w in sc["weeks"] if w["season"] == season and w["week"] == week), None)
    if wk is None:
        return ""
    names = sc.get("source_names", SOURCES)
    rows = pd.DataFrame(wk["rows"])
    r = rows[(rows.fmt == "ppr") & rows.group.isin(TOP_N)].groupby("source")[["pair", "close", "lost"]].mean()
    r = r.sort_values("pair", ascending=False)
    lines = ["## Scorecard (every source, last forecast before each kickoff)", "",
             "PPR, RB/WR/TE. Players recommended who didn't play count as 0.", "",
             "| Source | Start/sit accuracy | Close calls | Points lost per decision |", "|---|---|---|---|"]
    for src, x in r.iterrows():
        cl = "–" if pd.isna(x.close) else f"{x.close:.1%}"
        lines.append(f"| {names.get(src, src)} | {x.pair:.1%} | {cl} | {x.lost:.2f} |")
    summ = sc.get("summary", {}).get(str(season))
    if summ and summ["weeks"] > 1:
        lines += ["", f"Season to date ({summ['weeks']} weeks), start/sit accuracy, PPR:", ""]
        for src, v in sorted(summ["sources"].items(), key=lambda kv: -(kv[1]["ppr"]["pair"] or 0)):
            e = v["ppr"]
            ci = e.get("vs_ecr")
            extra = (f" (vs. experts {100 * ci['diff']:+.1f} pts, 95% range {100 * ci['lo']:+.1f} to "
                     f"{100 * ci['hi']:+.1f})") if ci and ci["diff"] is not None else ""
            lines.append(f"- {names.get(src, src)}: {e['pair']:.1%}{extra}")
    return "\n".join(lines)
