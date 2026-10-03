"""Walk-forward backtest against FantasyPros expert consensus (ECR).

For each test season we train only on earlier data, predict every week, and
compare against the Friday ECR snapshot for the same week.

Two information sets (see features.INFO_SETS) are evaluated separately:
  FRIDAY  injury report and reserve lists only. The fair comparison with the
          Friday ECR snapshot.
  SUNDAY  adds gameday inactives. Matches the live 11:45 ET run; it knows more
          than the Friday experts did, so it is not an equal-information contest.

Only players who were ranked by ECR *and* played are compared (we have no
prediction rows for players who didn't play). Historical game lines and
weather have no timestamps, and no historical player props exist, so the
Vegas prop blend is not backtested; the live scorecard covers it.
"""
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import SCORING
from .data import current_season, load_ecr
from .features import INFO_SETS, build_features
from .metrics import (bootstrap_diff, calibration_table, decision_metrics, h2h_prob, pairwise,
                      pairwise_strict, pinball, rank_blend)
from .model import QUANTILES, Distribution, Projector, add_distribution, oos_frame

log = logging.getLogger(__name__)
TOP_N = {"WR": 48, "RB": 36, "TE": 18}
SOURCES = ("model", "ecr", "naive", "blend")
SOURCE_NAMES = {"model": "Model", "ecr": "Experts (ECR)", "naive": "Recent average",
                "blend": "Model+ECR blend"}
WARMUP_SEASONS = 2  # extra earlier seasons predicted only to train the range model


def ecr_by_week(d) -> pd.DataFrame:
    """Weekly ECR keyed by (player_id, season, week): `ecr` from the position
    pages and `ecr_flex` from the overall offense page."""
    ecr = load_ecr().copy()
    ecr["scrape_date"] = pd.to_datetime(ecr.scrape_date)
    ids = d["ff_ids"][["fantasypros_id", "gsis_id"]].dropna()
    ids["fantasypros_id"] = ids.fantasypros_id.astype("Int64").astype(str)
    ecr["id"] = ecr["id"].astype(str)
    ecr = ecr.merge(ids, left_on="id", right_on="fantasypros_id")

    s = d["schedules"]
    s = s[s.game_type == "REG"].copy()
    s["gameday"] = pd.to_datetime(s.gameday)
    ends = s.groupby(["season", "week"]).gameday.max().reset_index()  # week's last game
    ends = ends.sort_values("gameday")
    ecr = ecr.sort_values("scrape_date")
    ecr = pd.merge_asof(ecr, ends.rename(columns={"gameday": "week_end"}),
                        left_on="scrape_date", right_on="week_end", direction="forward")
    ecr = ecr.rename(columns={"gsis_id": "player_id"})
    key = ["player_id", "season", "week"]
    pos = ecr[ecr.page_type != "weekly-op"].drop_duplicates(key, keep="last")[key + ["ecr"]]
    flex = ecr[(ecr.page_type == "weekly-op") & ecr.pos.isin(TOP_N)].drop_duplicates(key, keep="last")
    return pos.merge(flex[key + ["ecr"]].rename(columns={"ecr": "ecr_flex"}), on=key, how="outer")


def predict_walk_forward(df, seasons, retrain_every=6):
    """Point predictions for every week of `seasons`, each from a model trained
    only on games before that week (retraining every `retrain_every` weeks)."""
    hist = df[~df.live]
    preds, oos = [], []
    for season in seasons:
        for start in range(1, 19, retrain_every):
            train = hist[(hist.season < season) | ((hist.season == season) & (hist.week < start))]
            test = hist[(hist.season == season) & hist.week.between(start, start + retrain_every - 1)]
            if test.empty:
                continue
            log.info("  %s weeks %d-%d: train %d rows", season, start, start + retrain_every - 1, len(train))
            p = Projector().fit(train).predict(test)
            for fmt in SCORING:
                p[f"actual_{fmt}"] = test[f"fp_{fmt}"].to_numpy()
            # Recent-form baseline. EWMA is linear, so std = 2*half - ppr exactly.
            p["naive_ppr"] = test.r_fp_ppr.to_numpy()
            p["naive_half"] = test.r_fp_half.to_numpy()
            p["naive_std"] = 2 * p.naive_half - p.naive_ppr
            preds.append(p)
            oos.append(oos_frame(p, test))
    return pd.concat(preds), pd.concat(oos, ignore_index=True)


def run_info(d, info, seasons, retrain_every=6, warmup=WARMUP_SEASONS):
    """Backtest one information set. Returns (predictions, out-of-sample frame)."""
    log.info("building features (%s information)", info)
    df = build_features(d, info=info)
    hist = df[~df.live]
    all_seasons = tuple(range(seasons[0] - warmup, seasons[0])) + tuple(seasons)
    p, oos = predict_walk_forward(df, all_seasons, retrain_every)

    # Ranges, chronologically: each season's range model learns only from
    # out-of-sample errors in earlier seasons (warm-up seasons supply 2022's).
    parts = []
    for season in seasons:
        dist = Distribution().fit(oos[oos.season < season])
        sub = p[p.season == season].copy()
        parts.append(add_distribution(sub, hist.loc[sub.index], dist))
    p = pd.concat(parts)
    p["info"] = info
    return p, oos


def run(d, seasons=None, info_sets=INFO_SETS, retrain_every=6):
    """Defaults to the four most recent completed seasons (2022–2025 during 2026)."""
    if seasons is None:
        last = current_season() - 1
        seasons = tuple(range(last - 3, last + 1))
    ecr = ecr_by_week(d)
    preds, oos_by = [], {}
    for info in info_sets:
        p, oos = run_info(d, info, seasons, retrain_every)
        preds.append(p.merge(ecr, on=["player_id", "season", "week"], how="left"))
        oos_by[info] = oos
    p = pd.concat(preds, ignore_index=True)
    return p, score_groups(p), oos_by


# ---------------------------------------------------------------- scoring

def _universe(g):
    """Experts' top N at each position among players who played and were ranked."""
    return pd.concat([g[g.position == pos].nsmallest(n, "ecr") for pos, n in TOP_N.items()])


def score_groups(p):
    """One row per (info, season, week, group, format): decision metrics for each source.

    group is RB/WR/TE (same-position pairs) or FLEX (cross-position pairs only,
    ordered for ECR by the overall offense page).
    """
    rows = []
    for (info, season, week), wk in p[p.ecr.notna()].groupby(["info", "season", "week"]):
        u = _universe(wk)
        groups = [(pos, u[u.position == pos]) for pos in TOP_N]
        fx = u[u.ecr_flex.notna()]
        groups.append(("FLEX", fx))
        for grp, g in groups:
            if len(g) < 8:
                continue
            flex = grp == "FLEX"
            ecr_score = -(g.ecr_flex if flex else g.ecr).to_numpy()
            ecr_rank = pd.Series(-ecr_score).rank(method="first").to_numpy()
            for fmt in SCORING:
                actual = g[f"actual_{fmt}"].to_numpy()
                model = g[f"{fmt}_proj"].to_numpy()
                scores = {"model": model, "ecr": ecr_score,
                          "naive": g[f"naive_{fmt}"].fillna(0).to_numpy(),
                          "blend": rank_blend(model, ecr_score)}
                m = decision_metrics(scores, actual, ecr_rank,
                                     cross=g.position.to_numpy() if flex else None)
                row = {"info": info, "season": season, "week": week, "group": grp, "fmt": fmt,
                       "n": len(g), **m}
                if not flex:
                    row["model_strict"] = pairwise_strict(model, actual)
                    row["ecr_strict"] = pairwise_strict(ecr_score, actual)
                    row["model_rho"] = spearmanr(model, actual)[0]
                    row["ecr_rho"] = spearmanr(ecr_score, actual)[0]
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report

def _pct(x):
    return "   –  " if pd.isna(x) else f"{100 * x:5.1f}%"


def _ci(rows, a, b, scale=100, unit="pts"):
    mean, lo, hi = bootstrap_diff(rows, a, b)
    if pd.isna(mean):
        return "–"
    return f"{scale * mean:+.2f} [{scale * lo:+.2f}, {scale * hi:+.2f}]"


def _decision_table(w, metric, lower_better=False):
    """Rows RB/WR/TE/ALL/FLEX; columns per source plus differences vs ECR with 95% CIs."""
    pct = metric in ("pair", "close")
    fmtv = _pct if pct else (lambda x: "   –  " if pd.isna(x) else f"{x:6.2f}")
    scale = 100 if pct else 1
    head = f"{'':6}" + "".join(f"{SOURCE_NAMES[s]:>17}" for s in SOURCES) + \
           f"{'Model − ECR [95% CI]':>28}{'Blend − ECR [95% CI]':>28}"
    lines = [head]
    for grp in [*TOP_N, "ALL", "FLEX"]:
        r = w[w.group.isin(TOP_N)] if grp == "ALL" else w[w.group == grp]
        if r.empty:
            continue
        cells = "".join(f"{fmtv(r[f'{s}_{metric}'].mean()):>17}" for s in SOURCES)
        lines.append(f"{grp:6}{cells}{_ci(r, f'model_{metric}', f'ecr_{metric}', scale):>28}"
                     f"{_ci(r, f'blend_{metric}', f'ecr_{metric}', scale):>28}")
    note = "(lower is better)" if lower_better else ""
    return "\n".join(lines) + (f"\n{note}" if note else "")


def verdict(rows, a="model_pair", b="ecr_pair"):
    """Plain-English reading of a difference and its 95% interval."""
    mean, lo, hi = bootstrap_diff(rows, a, b)
    if lo > 0:
        return f"ahead of the experts ({100 * mean:+.2f} pts; the 95% interval excludes zero)"
    if hi < 0:
        return f"behind the experts ({100 * mean:+.2f} pts; the 95% interval excludes zero)"
    return f"statistically tied with the experts ({100 * mean:+.2f} pts, 95% interval {100 * lo:+.2f} to {100 * hi:+.2f})"


def _distribution_section(p, fmt):
    lines = []
    for label, sub in (("all player-games", p), (f"projected 8+ {fmt.upper()} pts", p[p[f"{fmt}_proj"] >= 8])):
        cov = " ".join(f"q{int(q * 100)} {np.mean(sub[f'actual_{fmt}'] < sub[f'{fmt}_q{int(q * 100)}']):.3f}"
                       for q in QUANTILES)
        pin = np.mean([pinball(sub[f"actual_{fmt}"], sub[f"{fmt}_q{int(q * 100)}"], q) for q in QUANTILES])
        inside = sub[f"actual_{fmt}"].between(sub[f"{fmt}_q10"], sub[f"{fmt}_q90"]).mean()
        width = (sub[f"{fmt}_q90"] - sub[f"{fmt}_q10"]).mean()
        lines.append(f"  {label} (n={len(sub)}): share below {cov}")
        lines.append(f"    10–90% range holds {inside:.1%} (target 80%), average width {width:.1f} pts, "
                     f"mean pinball loss {pin:.3f}")
    return lines


def h2h_pairs(p, fmt="ppr", max_pairs=60000):
    """(stated P(first scores more), first actually scored more) for same-position
    pairs in the experts' top N, using the page's method. Exact ties dropped."""
    rng = np.random.default_rng(0)
    qa, qb, wins = [], [], []
    qcols = [f"{fmt}_q{int(q * 100)}" for q in QUANTILES]
    for _, wk in p[p.ecr.notna()].groupby(["season", "week"]):
        u = _universe(wk)
        for pos in TOP_N:
            g = u[u.position == pos]
            i, j = np.triu_indices(len(g), 1)
            a = g[f"actual_{fmt}"].to_numpy()
            keep = a[i] != a[j]
            q = g[qcols].to_numpy()
            qa.append(q[i[keep]]), qb.append(q[j[keep]]), wins.append(a[i[keep]] > a[j[keep]])
    qa, qb, wins = np.concatenate(qa), np.concatenate(qb), np.concatenate(wins)
    if len(wins) > max_pairs:
        k = rng.choice(len(wins), max_pairs, replace=False)
        qa, qb, wins = qa[k], qb[k], wins[k]
    prob = np.concatenate([h2h_prob(qa[s:s + 5000], qb[s:s + 5000]) for s in range(0, len(qa), 5000)])
    return prob, wins


def fit_h2h_slope(p, fmt="ppr"):
    """Recalibration for head-to-head odds: P_shown = sigmoid(slope * logit(P_raw)).

    One parameter, no intercept (the pair order is arbitrary, so the map must be
    symmetric around 50%). slope < 1 pulls overconfident odds toward 50%.
    """
    from scipy.optimize import minimize_scalar
    prob, wins = h2h_pairs(p, fmt)
    z = np.log(np.clip(prob, 1e-4, 1 - 1e-4) / np.clip(1 - prob, 1e-4, 1))
    y = wins.astype(float)

    def nll(a):
        q = 1 / (1 + np.exp(-a * z))
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return float(minimize_scalar(nll, bounds=(0.2, 2.0), method="bounded").x)


def _h2h_section(p, fmt="ppr", max_pairs=60000):
    """Calibration of head-to-head probabilities for same-position pairs in the experts' top N."""
    prob, wins = h2h_pairs(p, fmt, max_pairs)
    slope = fit_h2h_slope(p, fmt)
    # Orient every pair so the stated probability is for the favorite.
    fav = prob >= 0.5
    pf, hit = np.where(fav, prob, 1 - prob), np.where(fav, wins, ~wins)
    cal = calibration_table(pf, hit.astype(float), [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0])
    brier = np.mean((pf - hit) ** 2)
    z = np.log(pf / (1 - pf))
    pc = 1 / (1 + np.exp(-slope * z))
    cal2 = calibration_table(pc, hit.astype(float), [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0])
    return [f"  {len(pf)} pairs, Brier score {brier:.4f} (lower is better; always saying 50% scores 0.25)",
            "  favorite's stated chance vs. how often the favorite actually scored more:",
            *("    " + ln for ln in cal.round(3).to_string().splitlines()),
            f"  after the page's correction (slope {slope:.3f}; fit on these same pairs, so in-sample), "
            f"Brier {np.mean((pc - hit) ** 2):.4f}:",
            *("    " + ln for ln in cal2.round(3).to_string().splitlines())]


def summarize(p, weekly):
    lines = [
        "BACKTEST: walk-forward, 2022–2025 regular seasons, vs. FantasyPros expert consensus (ECR)",
        "",
        "How to read this",
        "- Pairwise accuracy: for two players, how often the forecast put the higher scorer first.",
        "  Tied forecasts get half credit. Compared on the experts' top 48 WR / 36 RB / 18 TE",
        "  who played. ALL weights each position-week equally. FLEX uses cross-position pairs only.",
        "- Close decisions: pairs within 5 expert ranking places of each other.",
        "- Points lost: average points left on the bench per decision (wrong pick = the gap).",
        "- Model+ECR blend: equal-weight average of model rank and expert rank, fixed in advance.",
        "- [95% CI]: resampling whole weeks. If the interval includes 0, the difference is not",
        "  distinguishable from chance.",
        "- FRIDAY uses the injury report only (fair vs. Friday ECR). SUNDAY adds gameday inactives",
        "  (matches the live 11:45 ET run, so it knows more than the Friday experts did).",
        "- Not tested here: the Vegas prop blend (no historical props), and the timing of",
        "  historical game lines and weather (no timestamps). The live scorecard covers the blend.",
    ]
    for info in INFO_SETS:
        w = weekly[weekly["info"] == info]
        if w.empty:
            continue
        sub = p[p["info"] == info]
        ppr = w[w.fmt == "ppr"]
        lines += ["", "=" * 100, f"{info.upper()} information", "=" * 100, "",
                  "Headline (PPR, all positions): model is " + verdict(ppr[ppr.group.isin(TOP_N)]) + ".",
                  "Model+ECR blend is " + verdict(ppr[ppr.group.isin(TOP_N)], "blend_pair") + ".",
                  "", "Pairwise accuracy, PPR", _decision_table(ppr, "pair"),
                  "", "Close decisions (within 5 expert places), PPR", _decision_table(ppr, "close"),
                  "", "Points lost per decision, all pairs, PPR", _decision_table(ppr, "lost", True),
                  "", "Points lost per close decision, PPR", _decision_table(ppr, "close_lost", True),
                  "", "By scoring format (ALL positions): pairwise accuracy"]
        for fmt in SCORING:
            r = w[(w.fmt == fmt) & w.group.isin(TOP_N)]
            lines.append(f"  {fmt.upper():5} model {_pct(r.model_pair.mean())}  experts {_pct(r.ecr_pair.mean())}  "
                         f"recent avg {_pct(r.naive_pair.mean())}  blend {_pct(r.blend_pair.mean())}  "
                         f"model − experts {_ci(r, 'model_pair', 'ecr_pair')}")
        r = ppr[ppr.group.isin(TOP_N)]
        lines += ["", "By season (PPR, ALL positions): model vs experts"]
        for season, s in r.groupby("season"):
            lines.append(f"  {season}  model {_pct(s.model_pair.mean())}  experts {_pct(s.ecr_pair.mean())}  "
                         f"model wins {(s.groupby('week').model_pair.mean() > s.groupby('week').ecr_pair.mean()).mean():.0%} of weeks")
        lines += ["", "Old convention (tied forecasts count as wrong), PPR ALL, for comparison with earlier reports:",
                  f"  model {_pct(r.model_strict.mean())}  experts {_pct(r.ecr_strict.mean())}  "
                  f"(rank correlation: model {r.model_rho.mean():.3f}, experts {r.ecr_rho.mean():.3f})"]

        lines += ["", "Point accuracy (mean absolute error, all player-games; recent average shown for reference)"]
        for fmt in SCORING:
            for label, s in (("all", sub), ("proj 8+", sub[sub[f"{fmt}_proj"] >= 8])):
                mae = np.mean(np.abs(s[f"{fmt}_proj"] - s[f"actual_{fmt}"]))
                bias = np.mean(s[f"{fmt}_proj"] - s[f"actual_{fmt}"])
                nmae = np.mean(np.abs(s[f"naive_{fmt}"].fillna(0) - s[f"actual_{fmt}"]))
                lines.append(f"  {fmt.upper():5} {label:8} MAE {mae:5.2f}  bias {bias:+.2f}   recent avg MAE {nmae:5.2f}")

        lines += ["", "Ranges (chronological: each season's range model trained only on earlier seasons)"]
        for fmt in SCORING:
            lines.append(f" {fmt.upper()}")
            lines += _distribution_section(sub, fmt)

        lines += ["", "Boom % calibration (P of more than 20 points)"]
        for fmt in SCORING:
            cal = calibration_table(sub[f"{fmt}_boom"], (sub[f"actual_{fmt}"] > 20).astype(float),
                                    [0, .05, .1, .2, .3, .5, 1])
            lines += [f" {fmt.upper()}", *("   " + ln for ln in cal.round(3).to_string().splitlines())]

        lines += ["", "Head-to-head probability calibration (PPR, same page method, independence assumed)"]
        lines += _h2h_section(sub)
    return "\n".join(lines)
