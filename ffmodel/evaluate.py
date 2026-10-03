"""Score a finished week's frozen snapshots against actual results and expert consensus.

Reads snapshots/<season>/w<week>/{fri,sun}.json (see snapshot.py), writes a plain-English
report to reports/live/<season>_w<week>.md, keeps one row per snapshot and position in
reports/live/scores.csv, and rebuilds reports/live/SUMMARY.md (season to date).

Same rules as the backtest: only players who actually played count, and the expert
comparison uses the top N players by ECR at each position, so neither side gets credit
for knowing who was inactive. Players whose game kicked off before a snapshot was
taken are left out of that snapshot's scores.
"""
import json
from zoneinfo import ZoneInfo

import pandas as pd
from scipy.stats import spearmanr

from .backtest import TOP_N
from .metrics import pairwise as _pairwise
from .config import ROOT, SCORING, SNAPSHOT_DIR
from .features import STAT_MAP, score

LIVE_DIR = ROOT / "reports" / "live"
SCORES = LIVE_DIR / "scores.csv"
ET = ZoneInfo("America/New_York")
LABELS = {"fri": "Friday evening", "sun": "Sunday before kickoff"}
BACKTEST_REF = "2022–2025 backtest: us 61.8%, experts 61.6%"


def report_path(season, week):
    return LIVE_DIR / f"{season}_w{week:02d}.md"


def pending():
    """(season, week) pairs that have snapshots but no report yet. Standard library only."""
    out = []
    for wk in sorted(SNAPSHOT_DIR.glob("*/w*")):
        season, week = int(wk.parent.name), int(wk.name[1:])
        if any(wk.glob("*.json")) and not report_path(season, week).exists():
            out.append((season, week))
    return out


# ------------------------------------------------------------------ actual results
def week_final(d, season, week):
    """True once every game that week has a final score and appears in the stats feed."""
    s = d["schedules"]
    g = s[(s.season == season) & (s.week == week) & (s.game_type == "REG")]
    ps = d["player_stats"]
    have = set(ps[(ps.season == season) & (ps.week == week)].game_id)
    return len(g) > 0 and g.result.notna().all() and set(g.game_id) <= have


def actuals(d, season, week):
    """Fantasy points for every RB/WR/TE who played (snap or stat), all formats."""
    ps = d["player_stats"]
    ps = ps[(ps.season == season) & (ps.week == week) & (ps.season_type == "REG")].copy()
    ps["fumbles_lost"] = ps[["rushing_fumbles_lost", "receiving_fumbles_lost"]].fillna(0).sum(axis=1)
    ps["two_pt"] = ps[["rushing_2pt_conversions", "receiving_2pt_conversions"]].fillna(0).sum(axis=1)
    st = ps.rename(columns=STAT_MAP)[["player_id", "receptions", "rec_yards", "rec_tds",
                                     "rush_yards", "rush_tds", "fumbles_lost", "two_pt"]]
    pmap = d["players"][["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    sn = d["snaps"]
    sn = sn[(sn.season == season) & (sn.week == week) & (sn.offense_snaps > 0)]
    sn = sn.merge(pmap, left_on="pfr_player_id", right_on="pfr_id")[["gsis_id"]]
    a = st.merge(sn.rename(columns={"gsis_id": "player_id"}), on="player_id", how="outer")
    a = a.drop_duplicates("player_id").fillna(0)
    for fmt in SCORING:
        a[fmt] = score(a, fmt)
    return a[["player_id"] + list(SCORING)]


# ------------------------------------------------------------------ scoring
def _load(season, week, label):
    p = SNAPSHOT_DIR / str(season) / f"w{week:02d}" / f"{label}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _frame(snap, act):
    """Snapshot players joined to results, limited to games that started after the snapshot."""
    taken = pd.Timestamp(snap["meta"]["taken_at"]).tz_convert(ET).tz_localize(None)
    rows = [{"player_id": p["id"], "name": p["name"], "pos": p["pos"], "team": p["team"],
             "kickoff": pd.Timestamp(p["kickoff"]) if p.get("kickoff") else pd.NaT,
             **{f"{f}_proj": p[f]["proj"] for f in SCORING},
             **{f"{f}_q10": p[f]["q"][0] for f in SCORING},
             **{f"{f}_q90": p[f]["q"][-1] for f in SCORING},
             **{f"{f}_boom": p[f]["boom"] for f in SCORING}} for p in snap["players"]]
    df = pd.DataFrame(rows)
    started = df.kickoff <= taken
    ecr = pd.DataFrame(snap["ecr"]).dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    df = df[~started].merge(ecr[["gsis_id", "ecr"]], left_on="player_id", right_on="gsis_id", how="left")
    df = df.merge(act, on="player_id", how="inner", suffixes=("", "_act"))  # played only
    return df.rename(columns={f: f"{f}_act" for f in SCORING}), int(started.sum())


def score_snapshot(df):
    """Per-position rank accuracy vs experts (PPR) plus point accuracy."""
    rows = []
    for pos, n_top in TOP_N.items():
        g = df[(df.pos == pos) & df.ecr.notna()].nsmallest(n_top, "ecr")
        if len(g) < 8:
            continue
        rows.append({"position": pos, "n": len(g),
                     "model_pair": _pairwise(g.ppr_proj, g.ppr_act),
                     "ecr_pair": _pairwise(-g.ecr, g.ppr_act),
                     "model_rho": spearmanr(g.ppr_proj, g.ppr_act)[0],
                     "ecr_rho": spearmanr(-g.ecr, g.ppr_act)[0]})
    pos = pd.DataFrame(rows)
    if len(pos):
        all_row = pos[["model_pair", "ecr_pair", "model_rho", "ecr_rho"]].mean()
        pos = pd.concat([pos, pd.DataFrame([{"position": "ALL", "n": int(pos.n.sum()), **all_row}])],
                        ignore_index=True)
    pts = {}
    for f in SCORING:
        err = df[f"{f}_proj"] - df[f"{f}_act"]
        pts[f] = {"mae": err.abs().mean(), "bias": err.mean(),
                  "coverage": df[f"{f}_act"].between(df[f"{f}_q10"], df[f"{f}_q90"]).mean(),
                  "boom_pred": df[f"{f}_boom"].mean(), "boom_hit": (df[f"{f}_act"] > 20).mean()}
    return pos, pts


# ------------------------------------------------------------------ report
def _pct(x):
    return "–" if pd.isna(x) else f"{x:.1%}"


def _headline(pos):
    a = pos.set_index("position")
    if "ALL" not in a.index:
        return "not enough players to compare"
    won = [p for p in TOP_N if p in a.index and a.loc[p, "model_pair"] > a.loc[p, "ecr_pair"]]
    lost = [p for p in TOP_N if p in a.index and a.loc[p, "model_pair"] < a.loc[p, "ecr_pair"]]
    s = f"us {_pct(a.loc['ALL', 'model_pair'])} vs experts {_pct(a.loc['ALL', 'ecr_pair'])}"
    detail = []
    if won:
        detail.append("better at " + ", ".join(won))
    if lost:
        detail.append("worse at " + ", ".join(lost))
    return s + (f" ({'; '.join(detail)})" if detail else "")


def _season_table(season):
    if not SCORES.exists():
        return ""
    s = pd.read_csv(SCORES)
    s = s[s.season == season]
    lines = []
    for label in ("sun", "fri"):
        t = s[s.snapshot == label]
        if t.empty:
            continue
        weeks = t.week.nunique()
        per = t.groupby("position")[["model_pair", "ecr_pair"]].mean().reindex([*TOP_N, "ALL"]).dropna()
        wins = t[t.position == "ALL"].pipe(lambda x: int((x.model_pair > x.ecr_pair).sum()))
        lines += [f"**{LABELS[label]} rankings, {weeks} week{'s' if weeks != 1 else ''}** "
                  f"(we beat the experts in {wins} of {weeks})", "",
                  "| Position | Us | Experts |", "|---|---|---|"]
        lines += [f"| {p} | {_pct(r.model_pair)} | {_pct(r.ecr_pair)} |" for p, r in per.iterrows()]
        lines.append("")
    return "\n".join(lines)


def write_summary():
    if not SCORES.exists():
        return
    seasons = sorted(pd.read_csv(SCORES).season.unique(), reverse=True)
    body = ["# Live track record", "",
            "How often we ranked two players at the same position in the right order "
            "(the one ranked higher scored more PPR points), compared with FantasyPros expert consensus. "
            f"Season averages; one week on its own is mostly noise. For reference, {BACKTEST_REF}.", ""]
    for season in seasons:
        body += [f"## {season}", "", _season_table(season)]
    (LIVE_DIR / "SUMMARY.md").write_text("\n".join(body).rstrip() + "\n")


def evaluate_week(d, season, week):
    """Score every snapshot for the week; returns the report path (or None if nothing scorable)."""
    act = actuals(d, season, week)
    sections, score_rows, headlines, notes = [], [], [], []
    for label in ("sun", "fri"):
        snap = _load(season, week, label)
        if snap is None:
            notes.append(f"No {LABELS[label]} snapshot was taken this week.")
            continue
        df, n_started = _frame(snap, act)
        pos, pts = score_snapshot(df)
        if pos.empty:
            notes.append(f"{LABELS[label]} snapshot had too few players to score.")
            continue
        headlines.append(f"{LABELS[label]}: {_headline(pos)}")
        for r in pos.itertuples():
            score_rows.append({"season": season, "week": week, "snapshot": label, "position": r.position,
                               "n": r.n, "model_pair": r.model_pair, "ecr_pair": r.ecr_pair,
                               "model_rho": r.model_rho, "ecr_rho": r.ecr_rho,
                               "mae_ppr": pts["ppr"]["mae"], "bias_ppr": pts["ppr"]["bias"],
                               "coverage_ppr": pts["ppr"]["coverage"]})
        m = snap["meta"]
        taken = pd.Timestamp(m["taken_at"]).tz_convert(ET)
        lines = [f"## {LABELS[label]} rankings", "",
                 f"Taken {taken:%a %b %-d, %-I:%M %p} ET. Expert rankings dated {m.get('ecr_scrape_date')}.", "",
                 "| Position | Players | Us | Experts | |", "|---|---|---|---|---|"]
        for r in pos.itertuples():
            mark = "✅" if r.model_pair > r.ecr_pair else ("❌" if r.model_pair < r.ecr_pair else "=")
            lines.append(f"| {r.position} | {r.n} | {_pct(r.model_pair)} | {_pct(r.ecr_pair)} | {mark} |")
        lines += ["", "**Point accuracy** (all players who played)", "",
                  "| Format | Avg miss | Lean | In 10–90% range (aim: 80%) | Boom: predicted vs actual |",
                  "|---|---|---|---|---|"]
        for f, v in pts.items():
            lean = f"{abs(v['bias']):.1f} pts too {'high' if v['bias'] > 0 else 'low'}"
            lines.append(f"| {f.upper()} | {v['mae']:.1f} pts | {lean} | {_pct(v['coverage'])} | "
                         f"{_pct(v['boom_pred'])} vs {_pct(v['boom_hit'])} |")
        miss = df.assign(err=df.ppr_act - df.ppr_proj)
        fmt = lambda r: f"{r.name} ({r.pos}, {r.team}) projected {r.ppr_proj:.1f}, scored {r.ppr_act:.1f}"
        lines += ["", "**Biggest surprises (PPR)**", ""]
        lines += [f"- ⬆ {fmt(r)}" for r in miss.nlargest(3, "err").itertuples()]
        lines += [f"- ⬇ {fmt(r)}" for r in miss.nsmallest(3, "err").itertuples()]
        if n_started:
            lines += ["", f"_{n_started} players left out because their game had already started "
                          "when this snapshot was taken._"]
        sections.append("\n".join(lines))
    if not score_rows:
        return None

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(score_rows)
    if SCORES.exists():
        old = pd.read_csv(SCORES)
        old = old[~((old.season == season) & (old.week == week))]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["season", "week", "snapshot", "position"]).to_csv(SCORES, index=False, float_format="%.4f")
    write_summary()

    s = d["schedules"]
    last_week = int(s[(s.season == season) & (s.game_type == "REG")].week.max())
    title = f"# {season} Week {week} report: " + headlines[0].split(": ", 1)[1].split(" (")[0]
    body = [title, "",
            "**Rank accuracy:** pick any two players at the same position — how often did the one ranked "
            "higher actually score more PPR points? Compared on the experts' top "
            f"{TOP_N['WR']} WRs, {TOP_N['RB']} RBs and {TOP_N['TE']} TEs who played.", "",
            *[f"- {h}" for h in headlines], "", "\n\n".join(sections), "",
            "## Season to date", "", _season_table(season) or "First week scored.",
            f"_One week is mostly noise; judge on the season average. {BACKTEST_REF}._"]
    if week == last_week:
        body += ["", "## Season wrap-up", "",
                 f"That was the last regular-season week. Final {season} numbers are above and in "
                 "`reports/live/SUMMARY.md`. Rankings pause until about 9 days before next season's opener."]
    if notes:
        body += ["", "## Notes", "", *[f"- {n}" for n in notes]]
    out = report_path(season, week)
    out.write_text("\n".join(body).rstrip() + "\n")
    return out
