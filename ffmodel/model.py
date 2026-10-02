"""Model training and prediction.

Projection (the average outcome) is an equal blend of two gradient-boosted
models trained on the same features:

* Component models predict each stat (receptions, receiving yards, TDs,
  carries, ...). Scoring them works for any format and lets us blend each
  stat with Vegas player props.
* A direct model predicts fantasy points for each format.

The range (floor/ceiling, boom odds) comes from a separate Distribution model
fit on *out-of-sample* backtest predictions: given a projection and how
volatile a player's usage is, how spread out were real outcomes?
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from .config import SCORING
from .features import feature_columns, score

COMPONENTS = {
    "receptions": "poisson", "rec_yards": "poisson", "rec_tds": "poisson",
    "carries": "poisson", "rush_yards": "squared_error", "rush_tds": "poisson",
    "fumbles_lost": "poisson", "two_pt": "poisson",
}
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
BOOM = 20.0


def _hgb(loss):
    return HistGradientBoostingRegressor(
        loss=loss, learning_rate=0.03, max_iter=500, max_leaf_nodes=15,
        min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)


class Projector:
    def fit(self, df: pd.DataFrame):
        train = df[~df.live]
        self.features = feature_columns(train)
        X = train[self.features]
        w = 0.88 ** (train.season.max() - train.season)  # favor recent seasons
        self.comp = {}
        for target, loss in COMPONENTS.items():
            y = train[target].clip(lower=0) if loss == "poisson" else train[target]
            self.comp[target] = _hgb(loss).fit(X, y, sample_weight=w)
        self.direct = {fmt: _hgb("squared_error").fit(X, train[f"fp_{fmt}"], sample_weight=w)
                       for fmt in SCORING}
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features]
        out = df[["player_id", "season", "week", "team", "opp", "position"]].copy()
        for target in COMPONENTS:
            out[f"proj_{target}"] = np.clip(self.comp[target].predict(X), 0, None)
        comp = out.rename(columns={f"proj_{c}": c for c in COMPONENTS})
        for fmt in SCORING:
            out[f"{fmt}_direct"] = self.direct[fmt].predict(X)
            out[f"{fmt}_proj"] = (score(comp, fmt) + out[f"{fmt}_direct"]) / 2
        return out


# ---------------------------------------------------------------- distribution

DIST_FEATURES = ["proj", "pos_code", "cv", "l_adot", "td_share", "r_snap_pct"]


def dist_frame(proj: pd.Series, feats: pd.DataFrame, td_pts: pd.Series) -> pd.DataFrame:
    """Inputs for the distribution model: projection plus volatility signals."""
    return pd.DataFrame({
        "proj": proj.to_numpy(),
        "pos_code": feats.pos_code.to_numpy(),
        "cv": (feats.fp_vol / feats.r_fp_half.clip(lower=1)).to_numpy(),
        "l_adot": feats.l_adot.to_numpy(),
        "td_share": (td_pts / proj.clip(lower=1)).to_numpy(),
        "r_snap_pct": feats.r_snap_pct.to_numpy(),
    }).fillna(-1)


class Distribution:
    """Quantiles of fantasy points given a projection, per format.

    Models outcomes as a multiple of the projection, so ranges scale up
    naturally for the highest-projected players.
    """

    def fit(self, oos: pd.DataFrame):
        self.models = {}
        for fmt in SCORING:
            X = oos[[f"{c}_{fmt}" if c in ("proj", "td_share") else c for c in DIST_FEATURES]]
            X.columns = DIST_FEATURES
            for q in QUANTILES:
                ratio = oos[f"actual_{fmt}"] / X.proj.clip(lower=3)
                self.models[(fmt, q)] = GradientBoostingRegressor(
                    loss="quantile", alpha=q, n_estimators=200, max_depth=3, learning_rate=0.05,
                    min_samples_leaf=150, subsample=0.8, random_state=0).fit(X, ratio)
        return self

    def quantiles(self, X: pd.DataFrame, fmt: str) -> np.ndarray:
        base = X.proj.clip(lower=3).to_numpy()[:, None]
        qs = np.column_stack([self.models[(fmt, q)].predict(X[DIST_FEATURES]) for q in QUANTILES]) * base
        qs = np.maximum(qs, 0)
        return np.maximum.accumulate(qs, axis=1)  # enforce monotone quantiles


def oos_frame(pred: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """The columns the distribution model trains on, from backtest predictions."""
    o = pd.DataFrame({c: feats[c].to_numpy() for c in ("pos_code", "l_adot", "r_snap_pct")})
    o["cv"] = (feats.fp_vol / feats.r_fp_half.clip(lower=1)).to_numpy()
    td_pts = 6 * (pred.proj_rec_tds + pred.proj_rush_tds)
    for fmt in SCORING:
        o[f"proj_{fmt}"] = pred[f"{fmt}_proj"].to_numpy()
        o[f"td_share_{fmt}"] = (td_pts / pred[f"{fmt}_proj"].clip(lower=1)).to_numpy()
        o[f"actual_{fmt}"] = feats[f"fp_{fmt}"].to_numpy()
    o["season"] = feats.season.to_numpy()
    return o.fillna(-1)


def add_distribution(pred: pd.DataFrame, feats: pd.DataFrame, dist: Distribution) -> pd.DataFrame:
    td_pts = 6 * (pred.proj_rec_tds + pred.proj_rush_tds)
    for fmt in SCORING:
        X = dist_frame(pred[f"{fmt}_proj"], feats, td_pts)
        qs = dist.quantiles(X, fmt)
        for i, q in enumerate(QUANTILES):
            pred[f"{fmt}_q{int(q * 100)}"] = qs[:, i]
        pred[f"{fmt}_boom"] = prob_above(qs, BOOM)
    return pred


def prob_above(qs: np.ndarray, x: float) -> np.ndarray:
    """P(points > x) from quantiles, interpolating between them and using an
    exponential tail above the 90th percentile."""
    probs = np.array(QUANTILES)
    res = np.empty(len(qs))
    for i, row in enumerate(qs):
        if x <= row[0]:
            res[i] = 1 - probs[0] * x / max(row[0], 1e-6)
        elif x <= row[-1]:
            res[i] = 1 - np.interp(x, row, probs)
        else:
            scale = max((row[-1] - row[-2]) / np.log(0.25 / 0.10), 0.5)
            res[i] = 0.10 * np.exp(-(x - row[-1]) / scale)
    return np.clip(res, 0, 1)
