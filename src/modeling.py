from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


EPS = 1e-12


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def make_model(name, seed):
    if name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=500,
            max_features=0.70,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )

    if name == "histgb":
        return HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=8,
            l2_regularization=2.0,
            random_state=seed,
        )

    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_lambda=4.0,
            reg_alpha=0.1,
            min_child_weight=3,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=seed,
        )

    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=500,
            depth=5,
            learning_rate=0.035,
            loss_function="RMSE",
            l2_leaf_reg=5.0,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )

    raise ValueError(name)


class BlockPreprocessor:
    def __init__(self, pca_components=None):
        self.pca_components = pca_components
        self.keep_ = None
        self.var_keep_ = None
        self.imputer = None
        self.scaler = None
        self.pca = None
        self.out_names_ = None

    def fit(self, X, names):
        X = np.asarray(X, float)

        self.keep_ = np.isfinite(X).any(axis=0)
        if not self.keep_.any():
            self.keep_[0] = True

        X1 = X[:, self.keep_]
        names1 = [n for n, keep in zip(names, self.keep_) if keep]

        self.imputer = SimpleImputer(strategy="median")
        Z = self.imputer.fit_transform(X1)

        var = np.nanvar(Z, axis=0)
        self.var_keep_ = np.isfinite(var) & (var > 1e-12)

        if not self.var_keep_.any():
            self.var_keep_[0] = True

        Z = Z[:, self.var_keep_]
        names2 = [n for n, keep in zip(names1, self.var_keep_) if keep]

        self.scaler = StandardScaler()
        Z = self.scaler.fit_transform(Z)

        ncomp = self.pca_components
        total_var = float(np.sum(np.nanvar(Z, axis=0)))

        if (
            ncomp
            and np.isfinite(total_var)
            and total_var > 1e-10
            and Z.shape[0] >= 3
        ):
            ncomp = int(min(ncomp, Z.shape[1], Z.shape[0] - 1))
            if ncomp < Z.shape[1]:
                self.pca = PCA(n_components=ncomp, random_state=0)
                Z = self.pca.fit_transform(Z)
                self.out_names_ = [f"PC{i+1}" for i in range(Z.shape[1])]
            else:
                self.out_names_ = names2
        else:
            self.out_names_ = names2

        return self

    def transform(self, X):
        X = np.asarray(X, float)
        Z = X[:, self.keep_]
        Z = self.imputer.transform(Z)
        Z = Z[:, self.var_keep_]
        Z = self.scaler.transform(Z)
        if self.pca is not None:
            Z = self.pca.transform(Z)
        return np.asarray(Z, np.float32)


class CMGTPreprocessor:
    PROPOSED_BLOCKS = [
        "image_trajectory",
        "phenotype_trajectory",
        "environment_trajectory",
        "concordance",
        "meta_reliability",
    ]

    STATIC_BLOCKS = [
        "static_image",
        "static_phenotype",
        "static_environment",
    ]

    def __init__(self, cfg):
        self.cfg = cfg
        self.pp = {}

    def _pca_for(self, block):
        f = self.cfg["features"]
        return {
            "image_trajectory": f.get("image_pca_components", 48),
            "phenotype_trajectory": f.get("phenotype_pca_components", 32),
            "environment_trajectory": f.get("environment_pca_components", 24),
            "concordance": None,
            "meta_reliability": None,
            "static_image": f.get("static_image_pca_components", 32),
            "static_phenotype": f.get("static_phenotype_pca_components", 24),
            "static_environment": f.get("static_environment_pca_components", 16),
        }[block]

    def fit(self, data, train_idx):
        for block in self.PROPOSED_BLOCKS + self.STATIC_BLOCKS:
            pp = BlockPreprocessor(self._pca_for(block))
            pp.fit(
                data[block][train_idx],
                data["feature_names"][block],
            )
            self.pp[block] = pp
        return self

    def transform_block(self, data, idx, block):
        return self.pp[block].transform(data[block][idx])

    def proposed_matrix(self, data, idx, include=None):
        blocks = include or self.PROPOSED_BLOCKS
        return np.concatenate(
            [self.transform_block(data, idx, b) for b in blocks],
            axis=1,
        )

    def static_matrix(self, data, idx, include=None):
        blocks = include or self.STATIC_BLOCKS
        return np.concatenate(
            [self.transform_block(data, idx, b) for b in blocks],
            axis=1,
        )


def inner_cv(n, folds, seed):
    k = min(int(folds), max(2, n // 10))
    k = max(2, min(k, n))
    return KFold(n_splits=k, shuffle=True, random_state=seed)


class OOFBlendRegressor:
    """
    Computationally light final prediction head.

    Candidate heads are evaluated on identical OOF folds. A non-negative
    linear blend is accepted only when OOF RMSE improves over the best
    single candidate. Otherwise the strongest single candidate is retained.
    """

    def __init__(self, candidate_names, folds=5, seed=42, use_blend=True):
        self.candidate_names = list(candidate_names)
        self.folds = folds
        self.seed = seed
        self.use_blend = use_blend

    def fit(self, X, y):
        cv = inner_cv(len(y), self.folds, self.seed)

        candidates = []
        for i, name in enumerate(self.candidate_names):
            try:
                model = make_model(name, self.seed + i)
                oof = cross_val_predict(model, X, y, cv=cv, n_jobs=None)
                score = rmse(y, oof)
                fitted = clone(model).fit(X, y)
                candidates.append({
                    "name": name,
                    "oof": oof,
                    "rmse": score,
                    "model": fitted,
                })
            except Exception as e:
                print(f"[MODEL WARN] {name} skipped: {e}")

        if not candidates:
            raise RuntimeError("No candidate regression head succeeded.")

        candidates.sort(key=lambda d: d["rmse"])
        self.best_name_ = candidates[0]["name"]
        self.best_oof_rmse_ = candidates[0]["rmse"]
        self.models_ = candidates

        self.blend_ = None
        self.blend_oof_rmse_ = self.best_oof_rmse_

        if self.use_blend and len(candidates) >= 2:
            P = np.column_stack([d["oof"] for d in candidates])
            meta = LinearRegression(positive=True, fit_intercept=True)
            meta.fit(P, y)
            blend_oof = meta.predict(P)
            blend_score = rmse(y, blend_oof)

            if blend_score < self.best_oof_rmse_:
                self.blend_ = meta
                self.blend_oof_rmse_ = blend_score

        return self

    def predict(self, X):
        P = np.column_stack([d["model"].predict(X) for d in self.models_])
        if self.blend_ is not None:
            return self.blend_.predict(P)

        best_idx = next(i for i, d in enumerate(self.models_) if d["name"] == self.best_name_)
        return P[:, best_idx]

    def audit(self):
        return {
            "best_single": self.best_name_,
            "best_single_oof_rmse": self.best_oof_rmse_,
            "blend_used": self.blend_ is not None,
            "blend_oof_rmse": self.blend_oof_rmse_,
            "candidate_oof_rmse": {d["name"]: d["rmse"] for d in self.models_},
            "blend_coefficients": (
                self.blend_.coef_.tolist() if self.blend_ is not None else None
            ),
            "blend_intercept": (
                float(self.blend_.intercept_) if self.blend_ is not None else None
            ),
        }


def fit_extratrees(X, y, seed):
    return make_model("extratrees", seed).fit(X, y)


def fit_named(name, X, y, seed):
    return make_model(name, seed).fit(X, y)
