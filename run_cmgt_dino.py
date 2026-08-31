import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
import shutil
import pandas as pd
import joblib

from src.utils import (
    load_config,
    seed_everything,
    ensure_output_dirs,
    export_excel_safe,
)
from src.manifest import build_three_year_manifest
from src.backbones import build_embedding_cache
from src.sensors import SensorRepository
from src.trajectory_features import build_cmgt_samples
from src.evaluate import run_repeated_evaluation, run_loyo
from src.stats import statistical_tables
from src.plots import make_plots
from src.xai import block_permutation_importance, concordance_feature_importance


def ensure_manifest(cfg):
    target = Path(cfg["data"]["manifest_csv"])
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        for candidate in cfg["data"].get("external_manifest_candidates", []):
            p = Path(candidate)
            if p.exists():
                shutil.copy2(p, target)
                print("[MANIFEST] Reused:", p)
                break

    if not target.exists() and cfg["data"].get("auto_build_manifest", True):
        build_three_year_manifest(cfg)

    if not target.exists():
        raise FileNotFoundError(target)

    manifest = pd.read_csv(target)
    manifest["target"] = pd.to_numeric(manifest["target"], errors="coerce")
    manifest = manifest.dropna(subset=["target"]).reset_index(drop=True)

    actual = sorted(manifest["year"].dropna().astype(int).unique().tolist())
    expected = sorted(map(int, cfg["data"].get("years", [2023, 2024, 2025])))

    print("[MANIFEST] rows:", len(manifest))
    print("[MANIFEST] years:", actual)
    print("[MANIFEST] plant-year groups:", manifest["group"].nunique())

    if cfg["data"].get("strict_three_year", True) and actual != expected:
        raise RuntimeError(
            f"Strict three-year validation failed: expected={expected}, actual={actual}"
        )

    return manifest


def main():
    cfg = load_config("config.yaml")
    seed_everything(cfg["seed"])
    out = ensure_output_dirs(cfg["data"]["output_dir"])

    manifest = ensure_manifest(cfg)

    audit = pd.DataFrame([{
        "manifest_rows": len(manifest),
        "years": ";".join(map(str, sorted(manifest["year"].astype(int).unique()))),
        "plant_year_groups": manifest["group"].nunique(),
        "unique_dates": manifest["date"].nunique(),
        "target_min": manifest["target"].min(),
        "target_max": manifest["target"].max(),
    }])
    audit.to_csv(out["tables"] / "Table_1_Input_Audit.csv", index=False)
    print(audit.to_string(index=False))

    backbone = cfg["vision"]["proposed_backbone"]

    embeddings, embedding_meta = build_embedding_cache(
        manifest,
        cfg,
        backbone_name=backbone,
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

    data["meta"].to_csv(
        out["tables"] / "Table_2_Plant_Year_Sample_Audit.csv",
        index=False,
    )

    feature_audit = pd.DataFrame([
        {
            "block": block,
            "samples": data[block].shape[0],
            "features": data[block].shape[1],
            "finite_fraction": float(pd.notna(data[block]).mean()),
        }
        for block in [
            "image_trajectory",
            "phenotype_trajectory",
            "environment_trajectory",
            "concordance",
            "meta_reliability",
            "static_image",
            "static_phenotype",
            "static_environment",
        ]
    ])
    feature_audit.to_csv(
        out["tables"] / "Table_3_Feature_Block_Audit.csv",
        index=False,
    )

    metrics_df, predictions, runtime, ablation = run_repeated_evaluation(
        data,
        cfg,
        out,
    )

    metrics_df.to_csv(
        out["tables"] / "Table_4_All_Repeated_Metrics.csv",
        index=False,
    )

    summary = (
        metrics_df.groupby("model")
        .agg(
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
    summary.to_csv(
        out["tables"] / "Table_5_Model_Summary.csv",
        index=False,
    )

    predictions.to_csv(
        out["tables"] / "Prediction_Detail.csv",
        index=False,
    )
    runtime.to_csv(
        out["tables"] / "Table_6_Runtime.csv",
        index=False,
    )
    ablation.to_csv(
        out["tables"] / "Table_7_Ablation.csv",
        index=False,
    )

    friedman, wilcoxon = statistical_tables(metrics_df, cfg)
    friedman.to_csv(
        out["tables"] / "Table_8_Friedman.csv",
        index=False,
    )
    wilcoxon.to_csv(
        out["tables"] / "Table_9_Wilcoxon_Holm_EffectSize.csv",
        index=False,
    )

    loyo = pd.DataFrame()
    loyo_preds = pd.DataFrame()
    if cfg["experiment"].get("run_leave_one_year_out", True):
        loyo, loyo_preds = run_loyo(data, cfg)
        loyo.to_csv(
            out["tables"] / "Table_10_Leave_One_Year_Out.csv",
            index=False,
        )
        loyo_preds.to_csv(
            out["tables"] / "LOYO_Prediction_Detail.csv",
            index=False,
        )

    print("\n[RESULT] Core numerical tables saved before figures/XAI.")
    print(summary.to_string(index=False))

    try:
        make_plots(summary, ablation, predictions, loyo, out, cfg)
        print("[RESULT] Figures saved.")
    except Exception as e:
        print("[WARN] Plot generation failed; numerical results preserved:", repr(e))

    try:
        bundle = joblib.load(out["models"] / "CMGT_DINO_repeat0.joblib")
        block_imp = block_permutation_importance(bundle, data, cfg, out)
        conc_imp = concordance_feature_importance(bundle, data, cfg, out)
        print("[RESULT] XAI saved.")
    except Exception as e:
        block_imp = pd.DataFrame()
        conc_imp = pd.DataFrame()
        print("[WARN] XAI failed; numerical results preserved:", repr(e))

    errors = export_excel_safe(
        out["base"] / "CMGT_DINO_All_Results.xlsx",
        {
            "Input_Audit": audit,
            "Feature_Audit": feature_audit,
            "Model_Summary": summary,
            "Repeated_Metrics": metrics_df,
            "Ablation": ablation,
            "Friedman": friedman,
            "Wilcoxon_Holm": wilcoxon,
            "Runtime": runtime,
            "LOYO": loyo,
            "XAI_Block": block_imp,
            "XAI_Concordance": conc_imp,
        },
    )
    if errors:
        print("[WARN] Excel sheet errors:", errors)
    else:
        print("[RESULT] Excel workbook saved.")

    print("\n==============================================")
    print("CMGT-DINO experiment completed.")
    print("Results:", out["base"])
    print("==============================================")


if __name__ == "__main__":
    main()
