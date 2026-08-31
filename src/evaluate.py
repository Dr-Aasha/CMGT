from __future__ import annotations

import time
import numpy as np
import pandas as pd
import joblib

from .modeling import (
    CMGTPreprocessor,
    OOFBlendRegressor,
    fit_extratrees,
    fit_named,
)
from .utils import regression_metrics


def year_stratified_split(meta, test_fraction, seed):
    rng = np.random.default_rng(seed)
    test = []

    for year, idx_series in meta.groupby("year").groups.items():
        idx = np.asarray(list(idx_series), dtype=int)
        n_test = max(1, int(round(len(idx) * test_fraction)))
        chosen = rng.choice(idx, size=min(n_test, len(idx) - 1), replace=False)
        test.extend(chosen.tolist())

    test = np.asarray(sorted(set(test)), dtype=int)
    all_idx = np.arange(len(meta), dtype=int)
    train = np.setdiff1d(all_idx, test)

    return train, test


def _fit_proposed(data, pp, train_idx, cfg, seed, include_blocks=None):
    Xtr = pp.proposed_matrix(data, train_idx, include=include_blocks)
    model = OOFBlendRegressor(
        cfg["model"]["candidate_heads"],
        folds=cfg["model"]["inner_folds"],
        seed=seed,
        use_blend=cfg["model"].get("use_oof_blend", True),
    ).fit(Xtr, data["y"][train_idx])
    return model


def _baseline_predictions(data, pp, train_idx, test_idx, cfg, seed):
    ytr = data["y"][train_idx]
    out = {}

    # DINOv3 whole-season static image only.
    Xi_tr = pp.transform_block(data, train_idx, "static_image")
    Xi_te = pp.transform_block(data, test_idx, "static_image")
    out["DINOv3 Static Image ExtraTrees"] = (
        fit_extratrees(Xi_tr, ytr, seed).predict(Xi_te)
    )

    # Phenotype only.
    Xp_tr = pp.transform_block(data, train_idx, "static_phenotype")
    Xp_te = pp.transform_block(data, test_idx, "static_phenotype")
    out["Phenotype-only ExtraTrees"] = (
        fit_extratrees(Xp_tr, ytr, seed + 1).predict(Xp_te)
    )

    # Environment only.
    Xe_tr = pp.transform_block(data, train_idx, "static_environment")
    Xe_te = pp.transform_block(data, test_idx, "static_environment")
    out["Environment-only ExtraTrees"] = (
        fit_extratrees(Xe_tr, ytr, seed + 2).predict(Xe_te)
    )

    # Static multimodal early fusion.
    Xs_tr = pp.static_matrix(data, train_idx)
    Xs_te = pp.static_matrix(data, test_idx)

    out["Static Early Fusion ExtraTrees"] = (
        fit_extratrees(Xs_tr, ytr, seed + 3).predict(Xs_te)
    )

    for name, label in [
        ("xgboost", "Static Early Fusion XGBoost"),
        ("catboost", "Static Early Fusion CatBoost"),
        ("histgb", "Static Early Fusion HistGB"),
    ]:
        try:
            out[label] = fit_named(name, Xs_tr, ytr, seed + 10).predict(Xs_te)
        except Exception as e:
            print(f"[BASELINE WARN] {label} skipped: {e}")

    # Same DINO trajectory but without concordance.
    traj_blocks = [
        "image_trajectory",
        "phenotype_trajectory",
        "environment_trajectory",
        "meta_reliability",
    ]
    Xt_tr = pp.proposed_matrix(data, train_idx, include=traj_blocks)
    Xt_te = pp.proposed_matrix(data, test_idx, include=traj_blocks)
    out["Trajectory Fusion ExtraTrees"] = (
        fit_extratrees(Xt_tr, ytr, seed + 20).predict(Xt_te)
    )

    return out


def run_repeated_evaluation(data, cfg, outdirs):
    rows = []
    pred_rows = []
    runtime_rows = []
    ablation_rows = []

    repeats = int(cfg["experiment"]["repeats"])
    test_fraction = float(cfg["experiment"]["test_fraction"])

    for rep in range(repeats):
        seed = cfg["seed"] + rep

        train_idx, test_idx = year_stratified_split(
            data["meta"],
            test_fraction,
            seed,
        )

        pp = CMGTPreprocessor(cfg).fit(data, train_idx)

        ytr = data["y"][train_idx]
        yte = data["y"][test_idx]

        t0 = time.perf_counter()
        baselines = _baseline_predictions(
            data, pp, train_idx, test_idx, cfg, seed
        )
        runtime_rows.append({
            "repeat": rep,
            "model": "All baselines",
            "seconds": time.perf_counter() - t0,
        })

        # Proposed full model.
        t1 = time.perf_counter()
        model = _fit_proposed(data, pp, train_idx, cfg, seed)
        Xte = pp.proposed_matrix(data, test_idx)
        proposed = model.predict(Xte)
        runtime_rows.append({
            "repeat": rep,
            "model": "CMGT-DINO",
            "seconds": time.perf_counter() - t1,
        })

        predictions = {**baselines, "CMGT-DINO": proposed}

        for name, p in predictions.items():
            rows.append({
                "repeat": rep,
                "model": name,
                **regression_metrics(yte, p),
            })

            for j, idx in enumerate(test_idx):
                pred_rows.append({
                    "repeat": rep,
                    "model": name,
                    "row_index": int(idx),
                    "group": data["meta"].iloc[idx]["group"],
                    "year": int(data["meta"].iloc[idx]["year"]),
                    "y_true": float(yte[j]),
                    "y_pred": float(p[j]),
                })

        # Ablation with same adaptive head where possible.
        ablation_specs = {
            "Full CMGT-DINO": [
                "image_trajectory",
                "phenotype_trajectory",
                "environment_trajectory",
                "concordance",
                "meta_reliability",
            ],
            "Without Concordance": [
                "image_trajectory",
                "phenotype_trajectory",
                "environment_trajectory",
                "meta_reliability",
            ],
            "Without Environment Trajectory": [
                "image_trajectory",
                "phenotype_trajectory",
                "concordance",
                "meta_reliability",
            ],
            "Without Phenotype Trajectory": [
                "image_trajectory",
                "environment_trajectory",
                "concordance",
                "meta_reliability",
            ],
            "Image Trajectory Only": [
                "image_trajectory",
                "meta_reliability",
            ],
        }

        for label, blocks in ablation_specs.items():
            am = _fit_proposed(
                data, pp, train_idx, cfg, seed + 1000, include_blocks=blocks
            )
            Xab = pp.proposed_matrix(data, test_idx, include=blocks)
            ap = am.predict(Xab)

            ablation_rows.append({
                "repeat": rep,
                "variant": label,
                **regression_metrics(yte, ap),
            })

        if rep == 0:
            joblib.dump(
                {
                    "preprocessor": pp,
                    "model": model,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "model_audit": model.audit(),
                },
                outdirs["models"] / "CMGT_DINO_repeat0.joblib",
            )

        print(
            f"[EVAL] repeat {rep+1}/{repeats}: "
            f"CMGT-DINO RMSE={regression_metrics(yte, proposed)['RMSE']:.4f}; "
            f"head={model.audit()}"
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(pred_rows),
        pd.DataFrame(runtime_rows),
        pd.DataFrame(ablation_rows),
    )


def run_loyo(data, cfg):
    rows = []
    preds = []

    years = sorted(data["meta"]["year"].astype(int).unique().tolist())

    for heldout in years:
        test_idx = np.where(data["meta"]["year"].to_numpy(int) == heldout)[0]
        train_idx = np.where(data["meta"]["year"].to_numpy(int) != heldout)[0]

        if len(train_idx) < 10 or len(test_idx) < 2:
            continue

        seed = cfg["seed"] + 5000 + heldout
        pp = CMGTPreprocessor(cfg).fit(data, train_idx)

        yte = data["y"][test_idx]

        baselines = _baseline_predictions(
            data, pp, train_idx, test_idx, cfg, seed
        )

        model = _fit_proposed(data, pp, train_idx, cfg, seed)
        p = model.predict(pp.proposed_matrix(data, test_idx))

        allp = {**baselines, "CMGT-DINO": p}

        for name, pred in allp.items():
            rows.append({
                "heldout_year": heldout,
                "model": name,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                **regression_metrics(yte, pred),
            })

            for j, idx in enumerate(test_idx):
                preds.append({
                    "heldout_year": heldout,
                    "model": name,
                    "group": data["meta"].iloc[idx]["group"],
                    "y_true": float(yte[j]),
                    "y_pred": float(pred[j]),
                })

        print(
            f"[LOYO] heldout={heldout}; "
            f"CMGT-DINO RMSE={regression_metrics(yte, p)['RMSE']:.4f}"
        )

    return pd.DataFrame(rows), pd.DataFrame(preds)
