"""Cross-sectional return-ranking models.

All models share: ``fit(X, y, t)`` / ``predict(X) -> np.ndarray``; X are
cross-sectionally ranked features (NaN allowed), y the ranked residual return.

- ``LGBM``: gradient-boosted trees (LightGBM). Best-performing family in the
  crypto cross-section literature; early stopping on a purged inner split.
- ``RFRidge``: random Fourier features + ridge — the "virtue of complexity"
  estimator (Kelly, Malamud & Zhou 2024), a strong high-dimensional linear-family
  benchmark.
- ``TabPFNModel``: TabPFN (tabular foundation model). Needs the user's Prior Labs
  token (TABPFN_TOKEN) and its weights are non-commercial — optional.
- ``Ensemble``: average of per-date z-scored predictions.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
DAY = 86_400_000


def _inner_split(t: np.ndarray, frac: float = 0.15, gap_days: int = 3):
    days = np.unique(t)
    cut = days[int(len(days) * (1 - frac))]
    tr = t < cut - gap_days * DAY
    va = t >= cut
    return tr, va


class LGBM:
    name = "lgbm"

    def __init__(self, **params):
        self.params = {
            "objective": "regression", "learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 300,
            "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0,
            "n_estimators": 2000, "verbose": -1, "n_jobs": 8, "seed": 7,
        }
        self.params.update(params)
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, t: np.ndarray) -> "LGBM":
        import lightgbm as lgb

        tr, va = _inner_split(t)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        self.best_iter = self.model.best_iteration_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X, num_iteration=self.best_iter)

    def importance(self, cols) -> pd.Series:
        return pd.Series(self.model.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)


class RFRidge:
    name = "rfridge"

    def __init__(self, n_features: int = 4000, gamma: float = 2.0, alphas=(1e1, 1e2, 1e3, 1e4), seed: int = 11):
        self.P, self.gamma, self.alphas, self.seed = n_features, gamma, alphas, seed

    def _phi(self, X: np.ndarray) -> np.ndarray:
        Z = X @ self.W + self.b
        return np.sqrt(2.0 / self.P) * np.cos(Z)

    def fit(self, X: pd.DataFrame, y: np.ndarray, t: np.ndarray) -> "RFRidge":
        Xn = np.nan_to_num(np.asarray(X, dtype=np.float32))
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, self.gamma, size=(Xn.shape[1], self.P)).astype(np.float32)
        self.b = rng.uniform(0, 2 * np.pi, size=self.P).astype(np.float32)
        tr, va = _inner_split(t)
        F_tr, F_va = self._phi(Xn[tr]), self._phi(Xn[va])
        G = F_tr.T @ F_tr
        Fy = F_tr.T @ y[tr]
        best, best_corr = None, -np.inf
        for a in self.alphas:
            beta = np.linalg.solve(G + a * np.eye(self.P, dtype=np.float32), Fy)
            corr = np.corrcoef(F_va @ beta, y[va])[0, 1]
            if corr > best_corr:
                best, best_corr = a, corr
        F = self._phi(Xn)
        self.alpha = best
        self.beta = np.linalg.solve(F.T @ F + best * np.eye(self.P, dtype=np.float32), F.T @ y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._phi(np.nan_to_num(np.asarray(X, dtype=np.float32))) @ self.beta


class TabPFNModel:
    name = "tabpfn"

    def __init__(self, max_rows: int = 20_000, seed: int = 3):
        self.max_rows, self.seed = max_rows, seed

    @staticmethod
    def available() -> bool:
        if not os.environ.get("TABPFN_TOKEN"):
            return False
        try:
            import tabpfn  # noqa: F401
        except ImportError:
            return False
        return True

    def fit(self, X: pd.DataFrame, y: np.ndarray, t: np.ndarray) -> "TabPFNModel":
        from tabpfn import TabPFNRegressor

        rng = np.random.default_rng(self.seed)
        # most recent rows matter most: sample with recency weighting
        w = np.linspace(0.2, 1.0, len(y))
        idx = rng.choice(len(y), size=min(self.max_rows, len(y)), replace=False, p=w / w.sum())
        self.model = TabPFNRegressor()
        self.model.fit(np.asarray(X)[idx], y[idx])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(np.asarray(X))


def zscore_by_date(pred: np.ndarray, t: np.ndarray) -> np.ndarray:
    s = pd.Series(pred, index=t)
    g = s.groupby(level=0)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0).to_numpy()


MODELS = {"lgbm": LGBM, "rfridge": RFRidge, "tabpfn": TabPFNModel}
