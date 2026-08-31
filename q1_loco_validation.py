#!/usr/bin/env python3
"""
CMGT-DINOv3 — Leave-One-Image-Cohort-Out (LOCO) Validation
===========================================================

Purpose
-------
This script performs the final image-cohort-aware validation recommended for
the CMGT-DINOv3 manuscript.

A raw greenhouse RGB frame can be linked to several plant-year records.
Therefore plant-level random train/test splitting can place the identical
raw image in both partitions. LOCO solves this by treating every connected
image-sharing cohort as the indivisible validation unit:

    one complete image-sharing cohort -> test
    all remaining cohorts            -> train

With the current Horti-M3 alignment, the leakage audit found 18 image-sharing
cohorts:
    2023: 1 cohort
    2024: 9 cohorts
    2025: 8 cohorts

Every cohort is held out exactly once. Hence every plant-year target receives
one out-of-fold prediction, and identical raw image paths cannot occur in both
training and testing within a fold.

Experiments produced
--------------------
A. 18-fold LOCO main model comparison
   - DINOv3 Static Image ExtraTrees
   - Phenotype-only ExtraTrees
   - Environment-only ExtraTrees
   - Static Early Fusion ExtraTrees
   - Static Early Fusion XGBoost
   - Static Early Fusion CatBoost
   - Static Early Fusion HistGB
   - Trajectory Fusion ExtraTrees
   - CMGT-DINOv3

B. LOCO CMGT ablation
   - Full CMGT-DINOv3
   - Without Concordance
   - Without Environment Trajectory
   - Without Phenotype Trajectory
   - Image Trajectory Only

C. Optional LOCO visual-backbone ablation
   - EfficientNet-B0
   - ConvNeXt-Tiny
   - DINOv2-Small
   - DINOv3 ViT-S/16

   Each backbone is evaluated with:
   - Static Image ExtraTrees
   - Image Trajectory ExtraTrees
   - Full CMGT

Outputs
-------
cmgt_outputs/q1_validation/loco/
    Table_C0_Image_Cohort_Definitions.csv
    Table_C1_LOCO_Fold_Audit.csv
    Table_C2_LOCO_Model_Metrics_By_Fold.csv
    Table_C3_LOCO_Model_Macro_Summary.csv
    Table_C4_LOCO_Model_Pooled_Summary.csv
    Table_C5_LOCO_Paired_Statistics.csv
    Table_C6_LOCO_Ablation_By_Fold.csv
    Table_C7_LOCO_Ablation_Macro_Summary.csv
    Table_C8_LOCO_Ablation_Pooled_Summary.csv
    Table_C9_LOCO_Per_Year_Pooled_Performance.csv
    LOCO_Prediction_Detail.csv
    LOCO_Ablation_Prediction_Detail.csv
    Figure_C1_Image_Cohort_Sizes.png
    Figure_C2_LOCO_Pooled_Model_RMSE.png
    Figure_C3_LOCO_Fold_RMSE_Distribution.png
    Figure_C4_LOCO_Ablation_RMSE.png
    Figure_C5_LOCO_Per_Year_RMSE.png

If backbone ablation is enabled:
    backbone/
        Table_CB1_Backbone_Metrics_By_Fold.csv
        Table_CB2_Backbone_Macro_Summary.csv
        Table_CB3_Backbone_Pooled_Summary.csv
        Table_CB4_DINOv3_vs_Backbones_Paired_Statistics.csv
        Table_CB5_Backbone_Friedman.csv
        Backbone_LOCO_Prediction_Detail.csv
        Figure_CB1_Full_CMGT_Backbone_Pooled_RMSE.png
        Figure_CB2_Backbone_Protocols_Pooled_RMSE.png

Recommended placement
---------------------
Copy this file into:
    ~/PhD-Projects/Aasha_Christhuraj/CMGT_DINO_HortiM3_v1_0/

It expects q1_validation_common.py from the previous Q1 validation package
to be in the same directory.

Run
---
    source ~/PhD-Projects/Aasha_Christhuraj/env_ramf/bin/activate
    cd ~/PhD-Projects/Aasha_Christhuraj/CMGT_DINO_HortiM3_v1_0
    export MPLBACKEND=Agg

Full run including backbone ablation:
    python3 q1_loco_validation.py --with-backbones

Main LOCO + ablation only:
    python3 q1_loco_validation.py

Backbone-only after the main LOCO has already run:
    python3 q1_loco_validation.py --backbones-only

Notes
-----
- DINOv3/DINOv2 gated-model access must already be authorized.
- Existing embedding caches are reused.
- No numerical result is hard-coded or invented.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from q1_validation_common import (
    BACKBONE_DISPLAY,
    ensure_dir,
    save_csv,
    load_manifest,
    build_group_image_map,
    build_components,
    image_union,
    groups_to_indices,
    fit_full_cmgt,
    holm_adjust,
    cohen_dz,
    bootstrap_ci,
)

from src.utils import (
    load_config,
    seed_everything,
    regression_metrics,
)
from src.backbones import build_embedding_cache
from src.sensors import SensorRepository
from src.trajectory_features import build_cmgt_samples
from src.modeling import (
    CMGTPreprocessor,
    OOFBlendRegressor,
    fit_extratrees,
    fit_named,
)


DEFAULT_BACKBONES = [
    "efficientnet_b0",
    "convnext_tiny",
    "dinov2_small",
    "dinov3_vits16",
]


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def save_figure(fig, path, dpi=600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", path)


def normalize_model_name(name):
    return "CMGT-DINOv3" if name in {"CMGT-DINO", "Full CMGT-DINO"} else name


def pooled_summary(pred_df, group_cols=("model",)):
    """
    Compute metrics after pooling all LOCO out-of-fold predictions.

    Because each image cohort is held out exactly once, each plant-year should
    appear once for each model. These pooled metrics weight every plant-year
    equally, regardless of cohort size.
    """
    rows = []

    for keys, g in pred_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        m = regression_metrics(
            g["y_true"].to_numpy(float),
            g["y_pred"].to_numpy(float),
        )
        row.update(m)
        row["n_predictions"] = int(len(g))
        row["n_unique_groups"] = int(g["group"].nunique())
        rows.append(row)

    return pd.DataFrame(rows)


def macro_summary(metrics_df, category_col):
    """
    Equal-weight mean across the 18 held-out cohorts.
    Cohorts have unequal sample counts, so this differs from pooled metrics.
    """
    out = (
        metrics_df.groupby(category_col)
        .agg(
            folds=("fold", "nunique"),
            test_n_mean=("test_n", "mean"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            R2_mean=("R2", "mean"),
            NRMSE_mean=("NRMSE", "mean"),
            sMAPE_mean=("sMAPE", "mean"),
        )
        .reset_index()
        .sort_values("RMSE_mean")
    )
    return out


def paired_statistics(
    metrics_df,
    category_col,
    reference,
    metric="RMSE",
    seed=42,
):
    """
    Pairwise statistics across identical LOCO folds.

    The statistic is fold-level and therefore gives each image-sharing cohort
    equal weight. Pooled metrics are reported separately.
    """
    pivot = metrics_df.pivot(
        index="fold",
        columns=category_col,
        values=metric,
    )

    if reference not in pivot.columns:
        raise KeyError(
            f"Reference '{reference}' not found in {category_col}."
        )

    ref = pivot[reference]
    tmp = []

    for other in pivot.columns:
        if other == reference:
            continue

        paired = pd.concat(
            [ref, pivot[other]],
            axis=1,
            keys=["ref", "other"],
        ).dropna()

        d = (
            paired["ref"].to_numpy(float)
            - paired["other"].to_numpy(float)
        )

        if len(d) < 3:
            continue

        try:
            stat, p = wilcoxon(
                d,
                zero_method="wilcox",
                alternative="two-sided",
            )
        except Exception:
            stat, p = np.nan, 1.0

        lo, hi = bootstrap_ci(
            d,
            n=5000,
            seed=seed,
        )

        tmp.append({
            "reference": reference,
            "comparison": other,
            "n_paired_folds": int(len(d)),
            f"mean_{metric}_difference_reference_minus_comparison": float(np.mean(d)),
            "wilcoxon_W": stat,
            "p_value": float(p),
            "cohen_dz": cohen_dz(d),
            "bootstrap95_low": lo,
            "bootstrap95_high": hi,
        })

    if not tmp:
        return pd.DataFrame()

    hp = holm_adjust(
        [r["p_value"] for r in tmp]
    )

    for row, p_adj in zip(tmp, hp):
        row["holm_p"] = float(p_adj)
        row["significant_0.05"] = bool(p_adj < 0.05)

    return pd.DataFrame(tmp).sort_values("holm_p")


def friedman_table(metrics_df, category_col, metric="RMSE"):
    pivot = metrics_df.pivot(
        index="fold",
        columns=category_col,
        values=metric,
    ).dropna(axis=0, how="any")

    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return pd.DataFrame()

    stat, p = friedmanchisquare(
        *[
            pivot[c].to_numpy(float)
            for c in pivot.columns
        ]
    )

    return pd.DataFrame([{
        "metric": metric,
        "friedman_chi2": float(stat),
        "p_value": float(p),
        "n_folds": int(pivot.shape[0]),
        "n_methods": int(pivot.shape[1]),
    }])


# ---------------------------------------------------------------------
# LOCO fold construction
# ---------------------------------------------------------------------

def create_loco_folds(manifest, group_to_images, image_to_groups):
    """
    Connected image-sharing components are the indivisible validation units.
    """
    component_map, cohort_df = build_components(
        manifest,
        image_to_groups,
    )

    meta = (
        manifest[["group", "year", "target"]]
        .drop_duplicates("group")
        .copy()
    )
    meta["component_id"] = meta["group"].map(component_map)

    if meta["component_id"].isna().any():
        missing = meta.loc[
            meta["component_id"].isna(),
            "group",
        ].tolist()
        raise RuntimeError(
            "Some plant-year groups were not assigned to an image-sharing "
            f"cohort: {missing[:10]}"
        )

    # A component spanning years would complicate interpretation but would
    # still be valid as a leakage-safe fold. Record it explicitly.
    cohort_detail = (
        meta.groupby("component_id")
        .agg(
            n_groups=("group", "nunique"),
            n_years=("year", "nunique"),
            years=("year", lambda x: ";".join(map(str, sorted(set(map(int, x)))))),
            yield_mean=("target", "mean"),
            yield_std=("target", "std"),
            yield_min=("target", "min"),
            yield_max=("target", "max"),
        )
        .reset_index()
        .sort_values(["years", "component_id"])
        .reset_index(drop=True)
    )

    # Add image counts per cohort.
    image_counts = []
    for cid in cohort_detail["component_id"]:
        groups = meta.loc[
            meta["component_id"] == cid,
            "group",
        ].astype(str).tolist()
        images = image_union(groups, group_to_images)
        image_counts.append(len(images))
    cohort_detail["unique_images"] = image_counts

    all_groups = set(meta["group"].astype(str))
    folds = []
    audit_rows = []

    for fold_idx, cid in enumerate(cohort_detail["component_id"], start=1):
        test_groups = sorted(
            meta.loc[
                meta["component_id"] == cid,
                "group",
            ].astype(str).tolist()
        )
        train_groups = sorted(all_groups - set(test_groups))

        train_images = image_union(train_groups, group_to_images)
        test_images = image_union(test_groups, group_to_images)
        overlap = train_images & test_images

        years = sorted(
            meta.loc[
                meta["component_id"] == cid,
                "year",
            ].astype(int).unique().tolist()
        )

        folds.append({
            "fold": fold_idx,
            "component_id": cid,
            "train_groups": train_groups,
            "test_groups": test_groups,
            "test_years": years,
        })

        audit_rows.append({
            "fold": fold_idx,
            "component_id": cid,
            "test_years": ";".join(map(str, years)),
            "train_n": len(train_groups),
            "test_n": len(test_groups),
            "train_unique_images": len(train_images),
            "test_unique_images": len(test_images),
            "shared_image_paths": len(overlap),
            "test_image_overlap_fraction": (
                len(overlap) / max(len(test_images), 1)
            ),
            "zero_raw_image_overlap": len(overlap) == 0,
        })

    audit_df = pd.DataFrame(audit_rows)

    if not audit_df["zero_raw_image_overlap"].all():
        bad = audit_df.loc[
            ~audit_df["zero_raw_image_overlap"]
        ]
        raise RuntimeError(
            "LOCO construction failed: at least one fold still has raw-image "
            f"overlap.\n{bad}"
        )

    # Each plant-year must occur in the test set exactly once.
    test_occurrence = {}
    for s in folds:
        for g in s["test_groups"]:
            test_occurrence[g] = test_occurrence.get(g, 0) + 1

    bad_occurrence = {
        g: n
        for g, n in test_occurrence.items()
        if n != 1
    }

    missing_groups = sorted(all_groups - set(test_occurrence))

    if bad_occurrence or missing_groups:
        raise RuntimeError(
            "LOCO fold coverage is invalid. "
            f"bad_occurrence={bad_occurrence}; missing={missing_groups}"
        )

    return component_map, cohort_detail, folds, audit_df


# ---------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------

def fit_adaptive_cmgt(
    data,
    pp,
    train_idx,
    cfg,
    seed,
    blocks=None,
):
    Xtr = pp.proposed_matrix(
        data,
        train_idx,
        include=blocks,
    )

    model = OOFBlendRegressor(
        cfg["model"]["candidate_heads"],
        folds=cfg["model"]["inner_folds"],
        seed=seed,
        use_blend=cfg["model"].get("use_oof_blend", True),
    ).fit(
        Xtr,
        data["y"][train_idx],
    )
    return model


def baseline_predictions(
    data,
    pp,
    train_idx,
    test_idx,
    cfg,
    seed,
    image_label="DINOv3",
):
    ytr = data["y"][train_idx]
    out = {}

    # Static image.
    X0 = pp.transform_block(data, train_idx, "static_image")
    X1 = pp.transform_block(data, test_idx, "static_image")
    out[f"{image_label} Static Image ExtraTrees"] = (
        fit_extratrees(X0, ytr, seed)
        .predict(X1)
    )

    # Phenotype.
    X0 = pp.transform_block(data, train_idx, "static_phenotype")
    X1 = pp.transform_block(data, test_idx, "static_phenotype")
    out["Phenotype-only ExtraTrees"] = (
        fit_extratrees(X0, ytr, seed + 1)
        .predict(X1)
    )

    # Environment.
    X0 = pp.transform_block(data, train_idx, "static_environment")
    X1 = pp.transform_block(data, test_idx, "static_environment")
    out["Environment-only ExtraTrees"] = (
        fit_extratrees(X0, ytr, seed + 2)
        .predict(X1)
    )

    # Static early fusion.
    X0 = pp.static_matrix(data, train_idx)
    X1 = pp.static_matrix(data, test_idx)

    out["Static Early Fusion ExtraTrees"] = (
        fit_extratrees(X0, ytr, seed + 3)
        .predict(X1)
    )

    for internal_name, display in [
        ("xgboost", "Static Early Fusion XGBoost"),
        ("catboost", "Static Early Fusion CatBoost"),
        ("histgb", "Static Early Fusion HistGB"),
    ]:
        try:
            out[display] = (
                fit_named(
                    internal_name,
                    X0,
                    ytr,
                    seed + 10,
                )
                .predict(X1)
            )
        except Exception as e:
            print(
                f"[WARN] {display} skipped: {e}"
            )

    # Trajectory baseline without concordance/adaptive head.
    traj_blocks = [
        "image_trajectory",
        "phenotype_trajectory",
        "environment_trajectory",
        "meta_reliability",
    ]
    X0 = pp.proposed_matrix(
        data,
        train_idx,
        include=traj_blocks,
    )
    X1 = pp.proposed_matrix(
        data,
        test_idx,
        include=traj_blocks,
    )

    out["Trajectory Fusion ExtraTrees"] = (
        fit_extratrees(
            X0,
            ytr,
            seed + 20,
        )
        .predict(X1)
    )

    return out


# ---------------------------------------------------------------------
# Main DINOv3 LOCO model comparison
# ---------------------------------------------------------------------

def run_main_loco(
    data,
    folds,
    cfg,
    output_dir,
):
    metric_rows = []
    pred_rows = []
    runtime_rows = []
    ablation_rows = []
    ablation_pred_rows = []

    ablation_specs = {
        "Full CMGT-DINOv3": [
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

    for s in folds:
        fold = int(s["fold"])
        seed = int(cfg["seed"]) + 700000 + fold

        tr = groups_to_indices(
            data,
            s["train_groups"],
        )
        te = groups_to_indices(
            data,
            s["test_groups"],
        )

        if len(te) == 0:
            raise RuntimeError(
                f"LOCO fold {fold} has no test samples."
            )

        pp = CMGTPreprocessor(cfg).fit(data, tr)
        ytr = data["y"][tr]
        yte = data["y"][te]

        # Baselines.
        t0 = time.perf_counter()
        preds = baseline_predictions(
            data,
            pp,
            tr,
            te,
            cfg,
            seed,
            image_label="DINOv3",
        )
        runtime_rows.append({
            "fold": fold,
            "component_id": s["component_id"],
            "model": "All baselines",
            "seconds": time.perf_counter() - t0,
        })

        # Proposed full model.
        t1 = time.perf_counter()
        full_model = fit_adaptive_cmgt(
            data,
            pp,
            tr,
            cfg,
            seed + 100,
            blocks=None,
        )
        full_pred = full_model.predict(
            pp.proposed_matrix(data, te)
        )
        runtime_rows.append({
            "fold": fold,
            "component_id": s["component_id"],
            "model": "CMGT-DINOv3",
            "seconds": time.perf_counter() - t1,
        })

        preds["CMGT-DINOv3"] = full_pred

        # Main metrics / pooled prediction rows.
        for model_name, p in preds.items():
            m = regression_metrics(yte, p)

            metric_rows.append({
                "fold": fold,
                "component_id": s["component_id"],
                "test_years": ";".join(map(str, s["test_years"])),
                "train_n": int(len(tr)),
                "test_n": int(len(te)),
                "model": model_name,
                **m,
            })

            for j, idx in enumerate(te):
                pred_rows.append({
                    "fold": fold,
                    "component_id": s["component_id"],
                    "model": model_name,
                    "row_index": int(idx),
                    "group": str(data["meta"].iloc[idx]["group"]),
                    "year": int(data["meta"].iloc[idx]["year"]),
                    "y_true": float(yte[j]),
                    "y_pred": float(p[j]),
                    "error": float(p[j] - yte[j]),
                    "abs_error": float(abs(p[j] - yte[j])),
                })

        # Ablation: same adaptive OOF head for each variant.
        for variant, blocks in ablation_specs.items():
            model = fit_adaptive_cmgt(
                data,
                pp,
                tr,
                cfg,
                seed + 1000,
                blocks=blocks,
            )
            p = model.predict(
                pp.proposed_matrix(
                    data,
                    te,
                    include=blocks,
                )
            )

            m = regression_metrics(yte, p)

            ablation_rows.append({
                "fold": fold,
                "component_id": s["component_id"],
                "test_years": ";".join(map(str, s["test_years"])),
                "train_n": int(len(tr)),
                "test_n": int(len(te)),
                "variant": variant,
                **m,
            })

            for j, idx in enumerate(te):
                ablation_pred_rows.append({
                    "fold": fold,
                    "component_id": s["component_id"],
                    "variant": variant,
                    "group": str(data["meta"].iloc[idx]["group"]),
                    "year": int(data["meta"].iloc[idx]["year"]),
                    "y_true": float(yte[j]),
                    "y_pred": float(p[j]),
                })

        cm = regression_metrics(
            yte,
            preds["CMGT-DINOv3"],
        )

        print(
            f"[LOCO] fold {fold:02d}/{len(folds)} "
            f"cohort={s['component_id']} "
            f"test_n={len(te)} "
            f"years={s['test_years']} "
            f"CMGT RMSE={cm['RMSE']:.4f}"
        )

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.DataFrame(pred_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_preds_df = pd.DataFrame(ablation_pred_rows)

    # Confirm every group has exactly one prediction for every model.
    coverage = (
        preds_df.groupby("model")
        .agg(
            rows=("group", "size"),
            unique_groups=("group", "nunique"),
        )
        .reset_index()
    )

    expected = int(data["meta"]["group"].nunique())
    if not (
        (coverage["rows"] == expected)
        & (coverage["unique_groups"] == expected)
    ).all():
        raise RuntimeError(
            "LOCO pooled-prediction coverage failed.\n"
            + coverage.to_string(index=False)
        )

    macro = macro_summary(
        metrics_df,
        "model",
    )
    pooled = pooled_summary(
        preds_df,
        ("model",),
    ).sort_values("RMSE")

    stats = paired_statistics(
        metrics_df,
        "model",
        reference="CMGT-DINOv3",
        metric="RMSE",
        seed=cfg["seed"],
    )

    friedman = friedman_table(
        metrics_df,
        "model",
        metric="RMSE",
    )

    ab_macro = macro_summary(
        ablation_df,
        "variant",
    )
    ab_pooled = pooled_summary(
        ablation_preds_df.rename(
            columns={"variant": "model"}
        ),
        ("model",),
    ).rename(
        columns={"model": "variant"}
    ).sort_values("RMSE")

    ab_stats = paired_statistics(
        ablation_df,
        "variant",
        reference="Full CMGT-DINOv3",
        metric="RMSE",
        seed=cfg["seed"] + 1,
    )

    # Per-year pooled model metrics.
    per_year = pooled_summary(
        preds_df,
        ("model", "year"),
    ).sort_values(["year", "RMSE"])

    save_csv(
        metrics_df,
        output_dir / "Table_C2_LOCO_Model_Metrics_By_Fold.csv",
    )
    save_csv(
        macro,
        output_dir / "Table_C3_LOCO_Model_Macro_Summary.csv",
    )
    save_csv(
        pooled,
        output_dir / "Table_C4_LOCO_Model_Pooled_Summary.csv",
    )
    save_csv(
        stats,
        output_dir / "Table_C5_LOCO_Paired_Statistics.csv",
    )
    save_csv(
        friedman,
        output_dir / "Table_C5b_LOCO_Friedman.csv",
    )
    save_csv(
        ablation_df,
        output_dir / "Table_C6_LOCO_Ablation_By_Fold.csv",
    )
    save_csv(
        ab_macro,
        output_dir / "Table_C7_LOCO_Ablation_Macro_Summary.csv",
    )
    save_csv(
        ab_pooled,
        output_dir / "Table_C8_LOCO_Ablation_Pooled_Summary.csv",
    )
    save_csv(
        ab_stats,
        output_dir / "Table_C8b_LOCO_Ablation_Paired_Statistics.csv",
    )
    save_csv(
        per_year,
        output_dir / "Table_C9_LOCO_Per_Year_Pooled_Performance.csv",
    )
    save_csv(
        runtime_df,
        output_dir / "Table_C10_LOCO_Runtime.csv",
    )
    save_csv(
        coverage,
        output_dir / "Table_C11_LOCO_Prediction_Coverage.csv",
    )
    save_csv(
        preds_df,
        output_dir / "LOCO_Prediction_Detail.csv",
    )
    save_csv(
        ablation_preds_df,
        output_dir / "LOCO_Ablation_Prediction_Detail.csv",
    )

    # Figure: pooled model RMSE.
    q = pooled.sort_values("RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.barh(
        q["model"],
        q["RMSE"],
    )
    ax.set_xlabel("Pooled out-of-fold RMSE")
    ax.set_title(
        "Leave-One-Image-Cohort-Out model comparison"
    )
    ax.grid(axis="x", alpha=0.25)
    save_figure(
        fig,
        output_dir / "Figure_C2_LOCO_Pooled_Model_RMSE.png",
    )

    # Figure: fold RMSE distribution.
    models_order = (
        macro.sort_values("RMSE_mean")["model"].tolist()
    )
    arrays = [
        metrics_df.loc[
            metrics_df["model"] == m,
            "RMSE",
        ].to_numpy(float)
        for m in models_order
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.boxplot(
        arrays,
        tick_labels=models_order,
        showmeans=True,
    )
    ax.set_ylabel("Fold RMSE")
    ax.set_title(
        "RMSE distribution across 18 image-cohort holdouts"
    )
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.grid(axis="y", alpha=0.25)
    save_figure(
        fig,
        output_dir / "Figure_C3_LOCO_Fold_RMSE_Distribution.png",
    )

    # Figure: pooled ablation.
    q = ab_pooled.sort_values("RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        q["variant"],
        q["RMSE"],
    )
    ax.set_xlabel("Pooled out-of-fold RMSE")
    ax.set_title(
        "CMGT-DINOv3 LOCO ablation"
    )
    ax.grid(axis="x", alpha=0.25)
    save_figure(
        fig,
        output_dir / "Figure_C4_LOCO_Ablation_RMSE.png",
    )

    # Figure: per-year pooled RMSE for selected methods.
    selected = [
        "CMGT-DINOv3",
        "Trajectory Fusion ExtraTrees",
        "Static Early Fusion ExtraTrees",
        "Phenotype-only ExtraTrees",
        "Environment-only ExtraTrees",
    ]
    py = per_year[
        per_year["model"].isin(selected)
    ].copy()

    pivot = py.pivot(
        index="year",
        columns="model",
        values="RMSE",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x = np.arange(len(pivot.index))
    width = 0.15

    for j, col in enumerate(pivot.columns):
        ax.bar(
            x + (
                j - (len(pivot.columns) - 1) / 2
            ) * width,
            pivot[col].to_numpy(float),
            width=width,
            label=col,
        )

    ax.set_xticks(
        x,
        [str(v) for v in pivot.index],
    )
    ax.set_xlabel("Cultivation year")
    ax.set_ylabel("Pooled out-of-fold RMSE")
    ax.set_title(
        "LOCO performance by cultivation year"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    save_figure(
        fig,
        output_dir / "Figure_C5_LOCO_Per_Year_RMSE.png",
    )

    return {
        "metrics": metrics_df,
        "predictions": preds_df,
        "macro": macro,
        "pooled": pooled,
        "paired": stats,
        "friedman": friedman,
        "ablation": ablation_df,
        "ablation_pooled": ab_pooled,
        "per_year": per_year,
    }


# ---------------------------------------------------------------------
# LOCO backbone ablation
# ---------------------------------------------------------------------

def run_backbone_loco(
    manifest,
    folds,
    cfg,
    backbones,
    output_dir,
):
    output_dir = ensure_dir(output_dir)

    sensor_repo = SensorRepository(
        cfg["data"]["root"],
        cfg["data"]["sensor_features"],
    )

    metric_rows = []
    pred_rows = []

    for backbone in backbones:
        if backbone not in cfg["vision"]["backbones"]:
            raise KeyError(
                f"Backbone '{backbone}' is not defined in config.yaml."
            )

        display = BACKBONE_DISPLAY.get(
            backbone,
            backbone,
        )

        print("\n" + "=" * 78)
        print("[LOCO BACKBONE]", display)
        print("=" * 78)

        embeddings, embedding_meta = build_embedding_cache(
            manifest,
            cfg,
            backbone_name=backbone,
        )

        data = build_cmgt_samples(
            manifest,
            embeddings,
            embedding_meta,
            sensor_repo,
            cfg,
        )

        for s in folds:
            fold = int(s["fold"])
            seed = int(cfg["seed"]) + 900000 + fold

            tr = groups_to_indices(
                data,
                s["train_groups"],
            )
            te = groups_to_indices(
                data,
                s["test_groups"],
            )

            pp = CMGTPreprocessor(cfg).fit(
                data,
                tr,
            )

            ytr = data["y"][tr]
            yte = data["y"][te]

            protocols = {}

            # Static image.
            X0 = pp.transform_block(
                data,
                tr,
                "static_image",
            )
            X1 = pp.transform_block(
                data,
                te,
                "static_image",
            )
            protocols["Static Image ExtraTrees"] = (
                fit_extratrees(
                    X0,
                    ytr,
                    seed,
                )
                .predict(X1)
            )

            # Image trajectory.
            X0 = pp.transform_block(
                data,
                tr,
                "image_trajectory",
            )
            X1 = pp.transform_block(
                data,
                te,
                "image_trajectory",
            )
            protocols["Image Trajectory ExtraTrees"] = (
                fit_extratrees(
                    X0,
                    ytr,
                    seed + 1,
                )
                .predict(X1)
            )

            # Full CMGT.
            model = fit_adaptive_cmgt(
                data,
                pp,
                tr,
                cfg,
                seed + 2,
                blocks=None,
            )
            protocols["Full CMGT"] = model.predict(
                pp.proposed_matrix(data, te)
            )

            for protocol, p in protocols.items():
                m = regression_metrics(yte, p)

                metric_rows.append({
                    "backbone": display,
                    "backbone_key": backbone,
                    "embedding_dim": int(embeddings.shape[1]),
                    "protocol": protocol,
                    "fold": fold,
                    "component_id": s["component_id"],
                    "test_years": ";".join(map(str, s["test_years"])),
                    "train_n": int(len(tr)),
                    "test_n": int(len(te)),
                    **m,
                })

                for j, idx in enumerate(te):
                    pred_rows.append({
                        "backbone": display,
                        "backbone_key": backbone,
                        "protocol": protocol,
                        "fold": fold,
                        "component_id": s["component_id"],
                        "group": str(data["meta"].iloc[idx]["group"]),
                        "year": int(data["meta"].iloc[idx]["year"]),
                        "y_true": float(yte[j]),
                        "y_pred": float(p[j]),
                    })

            print(
                f"[LOCO BACKBONE] {display} "
                f"fold {fold:02d}/{len(folds)} "
                f"Full-CMGT RMSE="
                f"{regression_metrics(yte, protocols['Full CMGT'])['RMSE']:.4f}"
            )

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.DataFrame(pred_rows)

    macro = (
        metrics_df.groupby(
            ["protocol", "backbone", "backbone_key", "embedding_dim"]
        )
        .agg(
            folds=("fold", "nunique"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            R2_mean=("R2", "mean"),
            NRMSE_mean=("NRMSE", "mean"),
            sMAPE_mean=("sMAPE", "mean"),
        )
        .reset_index()
        .sort_values(["protocol", "RMSE_mean"])
    )

    pooled = pooled_summary(
        preds_df,
        ("protocol", "backbone", "backbone_key"),
    ).sort_values(["protocol", "RMSE"])

    full_metrics = metrics_df[
        metrics_df["protocol"] == "Full CMGT"
    ].copy()

    paired = paired_statistics(
        full_metrics,
        "backbone",
        reference="DINOv3 ViT-S/16",
        metric="RMSE",
        seed=cfg["seed"] + 33,
    )

    friedman = friedman_table(
        full_metrics,
        "backbone",
        metric="RMSE",
    )

    save_csv(
        metrics_df,
        output_dir / "Table_CB1_Backbone_Metrics_By_Fold.csv",
    )
    save_csv(
        macro,
        output_dir / "Table_CB2_Backbone_Macro_Summary.csv",
    )
    save_csv(
        pooled,
        output_dir / "Table_CB3_Backbone_Pooled_Summary.csv",
    )
    save_csv(
        paired,
        output_dir / "Table_CB4_DINOv3_vs_Backbones_Paired_Statistics.csv",
    )
    save_csv(
        friedman,
        output_dir / "Table_CB5_Backbone_Friedman.csv",
    )
    save_csv(
        preds_df,
        output_dir / "Backbone_LOCO_Prediction_Detail.csv",
    )

    # Full CMGT pooled backbone figure.
    q = pooled[
        pooled["protocol"] == "Full CMGT"
    ].sort_values("RMSE")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.barh(
        q["backbone"],
        q["RMSE"],
    )
    ax.set_xlabel("Pooled out-of-fold RMSE")
    ax.set_title(
        "LOCO backbone ablation: Full CMGT"
    )
    ax.grid(axis="x", alpha=0.25)
    save_figure(
        fig,
        output_dir / "Figure_CB1_Full_CMGT_Backbone_Pooled_RMSE.png",
    )

    # All protocols.
    protocols = [
        "Static Image ExtraTrees",
        "Image Trajectory ExtraTrees",
        "Full CMGT",
    ]
    back_order = [
        BACKBONE_DISPLAY.get(b, b)
        for b in backbones
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x = np.arange(len(back_order))
    width = 0.24

    for j, protocol in enumerate(protocols):
        q = (
            pooled[
                pooled["protocol"] == protocol
            ]
            .set_index("backbone")
            .reindex(back_order)
        )

        ax.bar(
            x + (j - 1) * width,
            q["RMSE"].to_numpy(float),
            width=width,
            label=protocol,
        )

    ax.set_xticks(
        x,
        back_order,
        rotation=15,
    )
    ax.set_ylabel("Pooled out-of-fold RMSE")
    ax.set_title(
        "LOCO visual-backbone comparison"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(
        fig,
        output_dir / "Figure_CB2_Backbone_Protocols_Pooled_RMSE.png",
    )

    return metrics_df, macro, pooled, paired, friedman


# ---------------------------------------------------------------------
# Cohort figures and interpretation report
# ---------------------------------------------------------------------

def save_cohort_figure(cohort_df, output_dir):
    q = cohort_df.copy()

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(
        np.arange(len(q)),
        q["n_groups"],
    )
    ax.set_xticks(
        np.arange(len(q)),
        [
            f"{cid}\n{yrs}"
            for cid, yrs in zip(
                q["component_id"],
                q["years"],
            )
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Plant-year groups in cohort")
    ax.set_title(
        "Image-sharing cohort sizes used for LOCO validation"
    )
    ax.grid(axis="y", alpha=0.25)

    save_figure(
        fig,
        output_dir / "Figure_C1_Image_Cohort_Sizes.png",
    )


def write_interpretation(
    output_dir,
    cohort_df,
    audit_df,
    main_results=None,
    backbone_results=None,
):
    lines = [
        "CMGT-DINOv3 — LEAVE-ONE-IMAGE-COHORT-OUT VALIDATION",
        "====================================================",
        "",
        f"Image-sharing cohorts: {len(cohort_df)}",
        f"Plant-year targets covered: {int(cohort_df['n_groups'].sum())}",
        f"Maximum cohort size: {int(cohort_df['n_groups'].max())}",
        f"Minimum cohort size: {int(cohort_df['n_groups'].min())}",
        f"Maximum shared raw image paths in any LOCO fold: "
        f"{int(audit_df['shared_image_paths'].max())}",
        "",
        "LOCO definition:",
        "  One complete image-sharing connected component is held out.",
        "  All remaining components are used for training.",
        "  Every plant-year target is tested exactly once.",
        "",
    ]

    if main_results is not None:
        pooled = main_results["pooled"]
        cmgt = pooled[
            pooled["model"] == "CMGT-DINOv3"
        ]
        if len(cmgt):
            r = cmgt.iloc[0]
            lines += [
                "Primary pooled CMGT-DINOv3 result:",
                f"  RMSE = {r['RMSE']:.6f}",
                f"  MAE = {r['MAE']:.6f}",
                f"  R2 = {r['R2']:.6f}",
                f"  NRMSE = {r['NRMSE']:.6f}",
                f"  sMAPE = {r['sMAPE']:.6f}",
                "",
            ]

        lines += [
            "Important reporting distinction:",
            "  Table_C3 is the equal-cohort macro mean across 18 folds.",
            "  Table_C4 is the pooled out-of-fold result across all plant-year",
            "  predictions and therefore weights every plant-year equally.",
            "  For a single overall predictive-performance value, Table_C4 is",
            "  usually the clearer headline result; Table_C3 shows sensitivity",
            "  to cohort-to-cohort variation.",
            "",
        ]

    if backbone_results is not None:
        pooled = backbone_results[2]
        q = pooled[
            pooled["protocol"] == "Full CMGT"
        ].sort_values("RMSE")
        if len(q):
            lines += [
                "Full-CMGT backbone pooled ranking:",
            ]
            for _, r in q.iterrows():
                lines.append(
                    f"  {r['backbone']}: RMSE={r['RMSE']:.6f}"
                )
            lines.append("")

    lines += [
        "Before manuscript revision, inspect:",
        "  Table_C1_LOCO_Fold_Audit.csv",
        "  Table_C4_LOCO_Model_Pooled_Summary.csv",
        "  Table_C5_LOCO_Paired_Statistics.csv",
        "  Table_C8_LOCO_Ablation_Pooled_Summary.csv",
        "  Table_C9_LOCO_Per_Year_Pooled_Performance.csv",
        "",
        "If backbone ablation was run, also inspect:",
        "  backbone/Table_CB3_Backbone_Pooled_Summary.csv",
        "  backbone/Table_CB4_DINOv3_vs_Backbones_Paired_Statistics.csv",
    ]

    (output_dir / "LOCO_Interpretation.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument(
        "--with-backbones",
        action="store_true",
        help="Also run the four-backbone comparison on identical LOCO folds.",
    )
    ap.add_argument(
        "--backbones-only",
        action="store_true",
        help="Skip main DINOv3 model/ablation and run only backbone LOCO.",
    )
    ap.add_argument(
        "--backbones",
        nargs="+",
        default=DEFAULT_BACKBONES,
        help="Backbone keys from config.yaml.",
    )

    args = ap.parse_args()

    cfg = load_config("config.yaml")
    seed_everything(cfg["seed"])

    manifest = load_manifest(cfg)

    output_dir = ensure_dir(
        Path(cfg["data"]["output_dir"])
        / "q1_validation"
        / "loco"
    )

    group_to_images, image_to_groups, unresolved = build_group_image_map(
        manifest,
        cfg,
    )

    component_map, cohort_df, folds, audit_df = create_loco_folds(
        manifest,
        group_to_images,
        image_to_groups,
    )

    save_csv(
        cohort_df,
        output_dir / "Table_C0_Image_Cohort_Definitions.csv",
    )
    save_csv(
        audit_df,
        output_dir / "Table_C1_LOCO_Fold_Audit.csv",
    )
    save_csv(
        unresolved,
        output_dir / "LOCO_Unresolved_Image_References.csv",
    )

    # Exact split assignments for full reproducibility.
    split_rows = []
    for s in folds:
        for role, gs in [
            ("train", s["train_groups"]),
            ("test", s["test_groups"]),
        ]:
            for g in gs:
                split_rows.append({
                    "fold": s["fold"],
                    "component_id": s["component_id"],
                    "role": role,
                    "group": g,
                })

    save_csv(
        pd.DataFrame(split_rows),
        output_dir / "Table_C1b_LOCO_Split_Assignments.csv",
    )

    save_cohort_figure(
        cohort_df,
        output_dir,
    )

    print("\n[LOCO AUDIT]")
    print(
        audit_df[
            [
                "fold",
                "component_id",
                "test_years",
                "test_n",
                "test_unique_images",
                "shared_image_paths",
            ]
        ].to_string(index=False)
    )

    if int(audit_df["shared_image_paths"].max()) != 0:
        raise RuntimeError(
            "LOCO audit failed: raw-image overlap is not zero."
        )

    main_results = None
    backbone_results = None

    if not args.backbones_only:
        print("\n" + "=" * 78)
        print("BUILDING DINOv3 FEATURES FOR MAIN LOCO VALIDATION")
        print("=" * 78)

        embeddings, embedding_meta = build_embedding_cache(
            manifest,
            cfg,
            backbone_name="dinov3_vits16",
        )

        sensor_repo = SensorRepository(
            cfg["data"]["root"],
            cfg["data"]["sensor_features"],
        )

        data = build_cmgt_samples(
            manifest,
            embeddings,
            embedding_meta,
            sensor_repo,
            cfg,
        )

        main_results = run_main_loco(
            data,
            folds,
            cfg,
            output_dir,
        )

    if args.with_backbones or args.backbones_only:
        print("\n" + "=" * 78)
        print("RUNNING LOCO BACKBONE ABLATION")
        print("=" * 78)

        backbone_results = run_backbone_loco(
            manifest,
            folds,
            cfg,
            args.backbones,
            output_dir / "backbone",
        )

    write_interpretation(
        output_dir,
        cohort_df,
        audit_df,
        main_results=main_results,
        backbone_results=backbone_results,
    )

    print("\n" + "=" * 78)
    print("LOCO VALIDATION COMPLETED")
    print("=" * 78)
    print("Output:", output_dir)
    print("\nMost important files:")

    if not args.backbones_only:
        print("  Table_C1_LOCO_Fold_Audit.csv")
        print("  Table_C4_LOCO_Model_Pooled_Summary.csv")
        print("  Table_C5_LOCO_Paired_Statistics.csv")
        print("  Table_C8_LOCO_Ablation_Pooled_Summary.csv")
        print("  Table_C9_LOCO_Per_Year_Pooled_Performance.csv")

    if args.with_backbones or args.backbones_only:
        print("  backbone/Table_CB3_Backbone_Pooled_Summary.csv")
        print("  backbone/Table_CB4_DINOv3_vs_Backbones_Paired_Statistics.csv")


if __name__ == "__main__":
    main()
