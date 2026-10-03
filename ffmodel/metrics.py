"""Evaluation metrics shared by the backtest and the live scorecard.

Start/sit decisions are scored as pairs: for two players, did the forecast put
the one who scored more first? A tied forecast gets half credit (it is a coin
flip, not a wrong answer). Points lost is the decision regret: when a pair is
ordered wrongly, the points the lower scorer left on the table.
"""
import numpy as np
import pandas as pd

from .model import QUANTILES

CLOSE_SPOTS = 5  # "close decision": players within this many ECR ranking places


def pair_index(n):
    return np.triu_indices(n, 1)


def pair_credit(score, actual, i, j):
    """Per-pair credit (1 right, 0.5 tied forecast, 0 wrong) and points lost,
    for pairs whose actual results differ. Returns (credit, lost, keep mask)."""
    s, a = np.asarray(score, float), np.asarray(actual, float)
    da, ds = a[i] - a[j], s[i] - s[j]
    keep = da != 0
    credit = np.where(ds == 0, 0.5, (np.sign(ds) == np.sign(da)).astype(float))
    lost = (1 - credit) * np.abs(da)
    return credit[keep], lost[keep], keep


def pairwise(score, actual):
    """Share of pairs ordered correctly, half credit for forecast ties."""
    i, j = pair_index(len(score))
    c, _, _ = pair_credit(score, actual, i, j)
    return c.mean() if len(c) else np.nan


def pairwise_strict(score, actual):
    """The pre-2026-10 convention: forecast ties count as wrong. Kept for continuity."""
    s, a = np.asarray(score), np.asarray(actual)
    i, j = pair_index(len(s))
    keep = a[i] != a[j]
    return np.mean(np.sign(s[i] - s[j])[keep] == np.sign(a[i] - a[j])[keep])


def decision_metrics(scores: dict, actual, ecr_rank=None, cross=None):
    """Pairwise accuracy and points lost for several sources over one group of players.

    scores: {source: array (higher = start)}; actual: points scored.
    ecr_rank: expert rank within the group, for close decisions (|diff| <= CLOSE_SPOTS).
    cross: optional array of labels; only pairs with different labels are used
           (cross-position flex decisions).
    """
    n = len(actual)
    i, j = pair_index(n)
    if cross is not None:
        cross = np.asarray(cross)
        m = cross[i] != cross[j]
        i, j = i[m], j[m]
    out = {}
    close = None
    if ecr_rank is not None:
        r = np.asarray(ecr_rank, float)
        close = np.abs(r[i] - r[j]) <= CLOSE_SPOTS
    for name, s in scores.items():
        c, lost, keep = pair_credit(s, actual, i, j)
        out[f"{name}_pair"] = c.mean() if len(c) else np.nan
        out[f"{name}_lost"] = lost.mean() if len(lost) else np.nan
        if close is not None:
            ck = close[keep]
            out[f"{name}_close"] = c[ck].mean() if ck.any() else np.nan
            out[f"{name}_close_lost"] = lost[ck].mean() if ck.any() else np.nan
    out["n_pairs"] = int((np.asarray(actual)[i] != np.asarray(actual)[j]).sum())
    if close is not None:
        out["n_close"] = int(close.sum())
    return out


def rank_blend(*scores):
    """Equal-weight average of ranks (higher = better). Chosen in advance as the
    model + expert consensus candidate; never tuned on results."""
    ranks = [pd.Series(np.asarray(s, float)).rank(ascending=False).to_numpy() for s in scores]
    return -np.mean(ranks, axis=0)


def bootstrap_diff(rows: pd.DataFrame, a: str, b: str, cluster=("season", "week"), reps=2000, seed=0):
    """Mean of rows[a] - rows[b] with a 95% interval from resampling whole weeks.

    Each row is one group-week (e.g. a position in a week); weeks are resampled
    with replacement so that same-week rows move together.
    """
    r = rows.dropna(subset=[a, b])
    if r.empty:
        return np.nan, np.nan, np.nan
    keys = r[list(cluster)].astype(str).agg("-".join, axis=1)
    codes, uniq = pd.factorize(keys)
    diff = (r[a] - r[b]).to_numpy()
    sums = np.bincount(codes, weights=diff)
    counts = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=reps)
    boots = (w @ sums) / (w @ counts)
    return diff.mean(), *np.percentile(boots, [2.5, 97.5])


# ---------------------------------------------------------------- distributions

def pinball(actual, q_pred, q):
    """Quantile (pinball) loss; lower is better. A proper score for quantile forecasts."""
    d = np.asarray(actual) - np.asarray(q_pred)
    return np.mean(np.maximum(q * d, (q - 1) * d))


def inverse_cdf(qs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse CDF from published quantiles, exactly as the page's sampler does it:
    linear between quantiles, linear down to a floor below q10, exponential tail above q90.

    qs: (n, 5) quantiles; u: (m,) probabilities. Returns (n, m).
    """
    P = np.array(QUANTILES)
    q = np.asarray(qs, float)
    lo = np.maximum(0, q[:, 0] - (q[:, 1] - q[:, 0]))
    scale = np.maximum((q[:, 4] - q[:, 3]) / np.log(2.5), 0.5)
    u = np.asarray(u)[None, :]
    out = np.empty((len(q), u.shape[1]))
    low, high = u < P[0], u > P[-1]
    out[:] = lo[:, None] + (q[:, [0]] - lo[:, None]) * u / P[0]
    mid = ~(low | high)
    for k in range(4):
        seg = mid & (u >= P[k]) & (u <= P[k + 1])
        t = (u - P[k]) / (P[k + 1] - P[k])
        val = q[:, [k]] + (q[:, [k + 1]] - q[:, [k]]) * t
        out = np.where(seg, val, out)
    tail = q[:, [4]] + scale[:, None] * np.log(0.1 / (1 - np.clip(u, None, 1 - 1e-9)))
    return np.where(high, tail, out)


def h2h_prob(qa: np.ndarray, qb: np.ndarray, grid=400) -> np.ndarray:
    """P(A scores more than B) under the page's independence assumption,
    computed on a deterministic probability grid instead of random draws."""
    u = (np.arange(grid) + 0.5) / grid
    xa, xb = inverse_cdf(qa, u), inverse_cdf(qb, u)
    # F_B(x) for each of A's grid values: share of B's grid values below it.
    xb.sort(axis=1)
    fb = np.stack([np.searchsorted(xb[k], xa[k], side="left") for k in range(len(xa))]) / grid
    return fb.mean(axis=1)


def calibration_table(pred, hit, bins):
    df = pd.DataFrame({"pred": pred, "hit": hit})
    g = df.groupby(pd.cut(df.pred, bins), observed=True)
    return g.agg(n=("pred", "size"), pred=("pred", "mean"), actual=("hit", "mean"))
