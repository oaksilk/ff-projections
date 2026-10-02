"""Walk-forward backtest against FantasyPros expert consensus (ECR).

For each test season we train only on earlier seasons, predict every week,
and compare rank accuracy against the Friday ECR snapshot for the same week.
Only players who were both ranked by ECR and actually played are compared,
so the model gets no credit for knowing about inactives.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import load_ecr
from .model import QUANTILES, Distribution, Projector, add_distribution, oos_frame

TOP_N = {"WR": 48, "RB": 36, "TE": 18}


def ecr_by_week(d) -> pd.DataFrame:
    ecr = load_ecr().copy()
    ecr["scrape_date"] = pd.to_datetime(ecr.scrape_date)
    ids = d["ff_ids"][["fantasypros_id", "gsis_id"]].dropna()
    ids["fantasypros_id"] = ids.fantasypros_id.astype("Int64").astype(str)
    ecr["id"] = ecr["id"].astype(str)
    ecr = ecr.merge(ids, left_on="id", right_on="fantasypros_id")

    s = d["schedules"]
    s = s[s.game_type == "REG"].copy()
    s["gameday"] = pd.to_datetime(s.gameday)
    sundays = s.groupby(["season", "week"]).gameday.max().reset_index()  # week's last game
    sundays = sundays.sort_values("gameday")
    ecr = ecr.sort_values("scrape_date")
    ecr = pd.merge_asof(ecr, sundays.rename(columns={"gameday": "week_end"}),
                        left_on="scrape_date", right_on="week_end", direction="forward")
    ecr = ecr.rename(columns={"gsis_id": "player_id"})
    return ecr[["player_id", "season", "week", "pos", "ecr"]].drop_duplicates(["player_id", "season", "week"])


def _pairwise(score, actual):
    """Share of player pairs where `score` orders them the same as `actual`."""
    s, a = np.asarray(score), np.asarray(actual)
    i, j = np.triu_indices(len(s), 1)
    keep = a[i] != a[j]
    return np.mean(np.sign(s[i] - s[j])[keep] == np.sign(a[i] - a[j])[keep])


def run(df, d, seasons=(2022, 2023, 2024, 2025), retrain_every=6):
    """Walk-forward: retrain every few weeks on everything before that week."""
    ecr = ecr_by_week(d)
    hist = df[~df.live]
    preds, oos = [], []
    for season in seasons:
        for start in range(1, 19, retrain_every):
            train = hist[(hist.season < season) | ((hist.season == season) & (hist.week < start))]
            test = hist[(hist.season == season) & hist.week.between(start, start + retrain_every - 1)]
            if test.empty:
                continue
            p = Projector().fit(train).predict(test)
            p["actual_ppr"] = test.fp_ppr.to_numpy()
            p["actual_half"] = test.fp_half.to_numpy()
            p["naive"] = test.r_fp_ppr.to_numpy()  # recent-form baseline
            preds.append(p)
            oos.append(oos_frame(p, test))
    p, oos = pd.concat(preds), pd.concat(oos, ignore_index=True)

    # Distribution calibration: fit on the other seasons, apply to this one.
    # Prediction rows keep the feature table's index, so we can look features up.
    parts = []
    for season in seasons:
        dist = Distribution().fit(oos[oos.season != season])
        sub = p[p.season == season].copy()
        parts.append(add_distribution(sub, hist.loc[sub.index], dist))
    p = pd.concat(parts)
    p = p.merge(ecr, on=["player_id", "season", "week"], how="left")

    rows = []
    for (season, week, pos), g in p[p.ecr.notna()].groupby(["season", "week", "position"]):
        g = g.nsmallest(TOP_N[pos], "ecr")
        if len(g) < 8:
            continue
        rows.append({
            "season": season, "week": week, "position": pos, "n": len(g),
            "model_rho": spearmanr(g.ppr_proj, g.actual_ppr)[0],
            "ecr_rho": spearmanr(-g.ecr, g.actual_ppr)[0],
            "naive_rho": spearmanr(g.naive.fillna(0), g.actual_ppr)[0],
            "model_pair": _pairwise(g.ppr_proj, g.actual_ppr),
            "ecr_pair": _pairwise(-g.ecr, g.actual_ppr),
            "blend_pair": _pairwise(_blend_rank(g), g.actual_ppr),
        })
    return p, pd.DataFrame(rows), oos


def _blend_rank(g):
    """50/50 average of model rank and ECR rank (higher = better)."""
    return -(g.ppr_proj.rank(ascending=False) + g.ecr.rank()) / 2


def summarize(p, weekly):
    lines = []
    s = weekly.groupby("position")[["model_rho", "ecr_rho", "naive_rho", "model_pair", "ecr_pair", "blend_pair"]].mean()
    s.loc["ALL"] = weekly[s.columns].mean()
    wins = weekly.assign(win=weekly.model_pair > weekly.ecr_pair).groupby("position").win.mean()
    s["model_beats_ecr_weeks"] = wins
    s.loc["ALL", "model_beats_ecr_weeks"] = (weekly.model_pair > weekly.ecr_pair).mean()
    lines.append("Rank accuracy vs actual PPR points (top-N by ECR, players who played)")
    lines.append(s.round(3).to_string())
    by_season = weekly.groupby("season")[["model_pair", "ecr_pair"]].mean().round(3)
    lines.append("\nPairwise accuracy by season\n" + by_season.to_string())

    lines.append("\nCalibration (all player-games, PPR):")
    for q in (10, 25, 50, 75, 90):
        lines.append(f"  actual below q{q}: {np.mean(p.actual_ppr < p[f'ppr_q{q}']):.3f}")
    for fmt in ("half", "ppr"):
        mae = np.mean(np.abs(p[f"{fmt}_proj"] - p[f"actual_{fmt}"]))
        bias = np.mean(p[f"{fmt}_proj"] - p[f"actual_{fmt}"])
        lines.append(f"  {fmt} MAE {mae:.2f}  bias {bias:+.2f}")
    bins = pd.cut(p.ppr_boom, [0, .05, .1, .2, .3, .5, 1])
    cal = p.groupby(bins, observed=True).apply(lambda g: pd.Series({
        "n": len(g), "pred": g.ppr_boom.mean(), "actual": (g.actual_ppr > 20).mean()}))
    lines.append("\nBoom (20+ PPR) calibration\n" + cal.round(3).to_string())
    return "\n".join(lines)
