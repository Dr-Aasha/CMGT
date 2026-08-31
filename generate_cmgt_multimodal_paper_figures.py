#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA

from src.utils import load_config, cosine
from src.image_paths import split_photo_raw, resolve_image, build_basename_index
from src.sensors import SensorRepository
from src.trajectory_features import (
    _split_stages,
    _image_stage_features,
    _phenotype_stage_features,
    _sensor_stage_features,
    _trajectory_signature,
)

STAGES = ("early", "middle", "late")
STAGE_LABELS = {"early": "Early", "middle": "Middle", "late": "Late"}
SIGNATURE_NAMES = [
    "E→M magnitude", "M→L magnitude", "E→L magnitude",
    "Acceleration", "Direction", "Early variability", "Late variability",
]
PREFERRED_PHENOTYPE_TOKENS = [
    "plant height", "stem diameter", "leaf area", "ndvi", "rvi", "lai", "lnc", "ldw"
]
PREFERRED_ENVIRONMENT = [
    "Air Temperature", "Relative Humidity", "Light Intensity", "CO2", "Soil Moisture", "VPD"
]


def safe_read_manifest(cfg):
    p = Path(cfg["data"]["manifest_csv"])
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}\nRun: python3 prepare_manifest.py")
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"]).copy()
    if "group" not in df.columns:
        df["group"] = df["year"].astype(int).astype(str) + "_" + df["plant_id"].astype(str)
    return df


def load_embedding_cache(cfg):
    backbone = cfg["vision"]["proposed_backbone"]
    cache = Path(cfg["data"]["cache_dir"]) / "vision" / backbone
    idx_path = cache / "embedding_index.csv"
    npz_path = cache / "embeddings.npz"
    if not idx_path.exists() or not npz_path.exists():
        raise FileNotFoundError(
            f"DINO cache missing under: {cache}\nRun: python3 run_cmgt_dino.py"
        )
    meta = pd.read_csv(idx_path)
    emb = np.load(npz_path)["embeddings"].astype(np.float32)
    if len(meta) != len(emb):
        raise RuntimeError(f"Embedding cache mismatch: index={len(meta)}, embeddings={len(emb)}")
    p2i = dict(zip(meta["image_path"].astype(str), meta["index"].astype(int)))
    return backbone, emb, meta, p2i


def tabular_columns(manifest):
    return [
        c for c in manifest.columns
        if c.startswith("tab__") and pd.to_numeric(manifest[c], errors="coerce").notna().any()
    ]


def choose_phenotype_columns(cols, max_n=6):
    chosen = []
    lower = {c: c.lower().replace("tab__", "") for c in cols}
    for token in PREFERRED_PHENOTYPE_TOKENS:
        for c, lc in lower.items():
            if token in lc and c not in chosen:
                chosen.append(c)
                break
        if len(chosen) >= max_n:
            return chosen
    for c in cols:
        if c not in chosen:
            chosen.append(c)
        if len(chosen) >= max_n:
            break
    return chosen


def pretty_feature_name(c):
    return str(c).replace("tab__", "").replace("_", " ").strip()


def count_resolved_images(group_df, cfg, basename_index):
    seen = set()
    for r in group_df.itertuples():
        for raw in split_photo_raw(getattr(r, "photo_raw", None)):
            p = resolve_image(raw, cfg["data"]["root"], getattr(r, "manual_csv", None), basename_index)
            if p:
                seen.add(p)
    return len(seen)


def representative_group(manifest, cfg, basename_index, year=None):
    df = manifest.copy()
    if year is not None:
        df = df[df["year"].astype(int) == int(year)]
    if df.empty:
        raise ValueError(f"No samples available for year={year}")

    year_median = df.groupby("group")["target"].median().median()
    tabs = tabular_columns(df)
    scored = []
    for group, g in df.groupby("group"):
        nimg = count_resolved_images(g, cfg, basename_index)
        if tabs:
            ph = g[tabs].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            phen_cov = float(np.isfinite(ph).mean())
        else:
            phen_cov = 0.0
        y = float(pd.to_numeric(g["target"], errors="coerce").median())
        distance = abs(y - year_median) / max(abs(year_median), 1.0)
        score = np.log1p(nimg) + 2.0 * phen_cov - 0.35 * distance
        scored.append((score, group))
    scored.sort(reverse=True)
    return scored[0][1]


def stage_representative_image(stage_df, cfg, basename_index):
    candidates = []
    for r in stage_df.sort_values("date").itertuples():
        date = getattr(r, "date", pd.NaT)
        for raw in split_photo_raw(getattr(r, "photo_raw", None)):
            p = resolve_image(raw, cfg["data"]["root"], getattr(r, "manual_csv", None), basename_index)
            if p and Path(p).exists():
                candidates.append((date, p))
    if not candidates:
        return None, pd.NaT
    candidates.sort(key=lambda x: pd.Timestamp.max if pd.isna(x[0]) else x[0])
    return candidates[len(candidates) // 2][1], candidates[len(candidates) // 2][0]


def show_image(ax, path, title):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=15, fontweight="bold")
    if path is None:
        ax.text(0.5, 0.5, "Image unavailable", ha="center", va="center", fontsize=13, transform=ax.transAxes)
        return
    ax.imshow(Image.open(path).convert("RGB"))


def normalize_stage_matrix(values):
    a = np.asarray(values, float)
    out = np.zeros_like(a, float)
    for i in range(a.shape[0]):
        row = a[i]
        ok = np.isfinite(row)
        if not ok.any():
            out[i] = np.nan
            continue
        mu = np.nanmean(row)
        sd = np.nanstd(row)
        out[i] = np.nan_to_num(row - mu, nan=0.0) if sd < 1e-10 else (row - mu) / sd
    return out


def build_selected_sample_features(group_df, cfg, emb, p2i, basename_index, sensor_repo):
    stages = _split_stages(group_df)
    tabs_all = tabular_columns(group_df)
    tabs_plot = choose_phenotype_columns(tabs_all, max_n=6)

    image_means, image_stds, image_counts = {}, {}, {}
    rep_images, rep_dates = {}, {}
    for stage in STAGES:
        mu, sd, count = _image_stage_features(
            stages[stage], emb, p2i, cfg["data"]["root"], basename_index
        )
        image_means[stage], image_stds[stage], image_counts[stage] = mu, sd, count
        p, d = stage_representative_image(stages[stage], cfg, basename_index)
        rep_images[stage], rep_dates[stage] = p, d

    image_sig = _trajectory_signature(
        image_means["early"], image_means["middle"], image_means["late"],
        early_var=float(np.nanmean(image_stds["early"])),
        late_var=float(np.nanmean(image_stds["late"])),
        normalize_vectors=True,
    )

    ph_stage_means_all, ph_stage_vars_all = {}, {}
    for stage in STAGES:
        _, means, var = _phenotype_stage_features(stages[stage], tabs_all)
        ph_stage_means_all[stage], ph_stage_vars_all[stage] = means, var

    pstack = np.vstack([ph_stage_means_all[s] for s in STAGES])
    ph_scale = np.nanmean(np.abs(pstack), axis=0) + np.nanstd(pstack, axis=0) + 1e-6
    ph_sig = _trajectory_signature(
        ph_stage_means_all["early"] / ph_scale,
        ph_stage_means_all["middle"] / ph_scale,
        ph_stage_means_all["late"] / ph_scale,
        early_var=ph_stage_vars_all["early"],
        late_var=ph_stage_vars_all["late"],
        normalize_vectors=False,
    )

    tab_to_idx = {c: i for i, c in enumerate(tabs_all)}
    ph_plot_values = np.asarray(
        [[ph_stage_means_all[s][tab_to_idx[c]] for s in STAGES] for c in tabs_plot],
        float,
    )

    env_dict, env_means, env_vars, env_modes = {}, {}, {}, []
    for stage in STAGES:
        frames, mode, _ = sensor_repo.frames_for_rows(stages[stage])
        d, means, var = _sensor_stage_features(frames, cfg)
        env_dict[stage], env_means[stage], env_vars[stage] = d, means, var
        env_modes.append(mode)

    estack = np.vstack([env_means[s] for s in STAGES])
    env_scale = np.nanmean(np.abs(estack), axis=0) + np.nanstd(estack, axis=0) + 1e-6
    env_sig = _trajectory_signature(
        env_means["early"] / env_scale,
        env_means["middle"] / env_scale,
        env_means["late"] / env_scale,
        early_var=env_vars["early"],
        late_var=env_vars["late"],
        normalize_vectors=False,
    )

    env_plot_names, env_plot_values = [], []
    for name in PREFERRED_ENVIRONMENT:
        vals = [env_dict[s].get(f"{name}__mean", np.nan) for s in STAGES]
        if np.isfinite(vals).any():
            env_plot_names.append(name)
            env_plot_values.append(vals)

    return {
        "stages": stages,
        "rep_images": rep_images,
        "rep_dates": rep_dates,
        "image_means": image_means,
        "image_counts": image_counts,
        "phenotype_names": [pretty_feature_name(c) for c in tabs_plot],
        "phenotype_values": ph_plot_values,
        "environment_names": env_plot_names,
        "environment_values": np.asarray(env_plot_values, float),
        "environment_modes": env_modes,
        "signatures": {"Image": image_sig, "Phenotype": ph_sig, "Environment": env_sig},
    }


def save_figure(fig, path, dpi=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", path)


def make_three_year_dataset_montage(manifest, cfg, basename_index, out_dir, dpi):
    years = sorted(manifest["year"].dropna().astype(int).unique())
    fig, axes = plt.subplots(len(years), 3, figsize=(14.5, 4.3 * len(years)), squeeze=False)

    for row, year in enumerate(years):
        group = representative_group(manifest, cfg, basename_index, year=year)
        g = manifest[manifest["group"] == group].sort_values("date")
        stages = _split_stages(g)
        y = float(g["target"].median())

        for col, stage in enumerate(STAGES):
            p, d = stage_representative_image(stages[stage], cfg, basename_index)
            date_text = pd.Timestamp(d).strftime("%Y-%m-%d") if not pd.isna(d) else "date unavailable"
            show_image(axes[row, col], p, f"{year} – {STAGE_LABELS[stage]} stage\n{date_text}")
            if col == 0:
                axes[row, col].set_ylabel(f"{group}\nYield = {y:.1f}", fontsize=14, fontweight="bold")

    fig.suptitle(
        "Representative Horti-M3 RGB observations across three cultivation years",
        fontsize=18, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.01,
        "Each row shows one automatically selected image-rich plant-year record. "
        "RGB observations are greenhouse canopy views associated with the plant record and may include neighboring plants.",
        ha="center", fontsize=11,
    )
    fig.subplots_adjust(hspace=0.24, wspace=0.05, bottom=0.06)
    save_figure(fig, out_dir / "Figure_Dataset_ThreeYear_Multimodal_Examples.png", dpi)


def make_multimodal_processing_figure(group, manifest, cfg, emb, p2i, basename_index, sensor_repo, out_dir, dpi):
    g = manifest[manifest["group"] == group].sort_values("date").copy()
    if g.empty:
        raise ValueError(f"Unknown group: {group}")

    feat = build_selected_sample_features(g, cfg, emb, p2i, basename_index, sensor_repo)
    y = float(g["target"].median())
    year = int(g["year"].iloc[0])

    # Global PCA coordinate system fitted on all cached DINOv3 image embeddings.
    pca = PCA(n_components=2, random_state=0)
    pca.fit(emb)
    stage_z = np.stack([feat["image_means"][s] for s in STAGES])
    coords = pca.transform(stage_z)

    fig = plt.figure(figsize=(16, 13))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.1, 1.0, 1.0], hspace=0.42, wspace=0.48)

    for j, stage in enumerate(STAGES):
        ax = fig.add_subplot(gs[0, 2*j:2*j+2])
        d = feat["rep_dates"][stage]
        dtext = pd.Timestamp(d).strftime("%Y-%m-%d") if not pd.isna(d) else "date unavailable"
        show_image(
            ax, feat["rep_images"][stage],
            f"{STAGE_LABELS[stage]} RGB\n{dtext} | n={feat['image_counts'][stage]} images",
        )

    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(coords[:, 0], coords[:, 1], marker="o", linewidth=2)
    for i, stage in enumerate(STAGES):
        ax.annotate(STAGE_LABELS[stage], (coords[i, 0], coords[i, 1]), xytext=(6, 6), textcoords="offset points", fontsize=12, fontweight="bold")
    ax.annotate("", xy=coords[1], xytext=coords[0], arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.annotate("", xy=coords[2], xytext=coords[1], arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.set_title("Frozen DINOv3 visual growth trajectory", fontsize=15, fontweight="bold")
    ax.set_xlabel("Global DINOv3 PCA component 1", fontsize=12)
    ax.set_ylabel("Global DINOv3 PCA component 2", fontsize=12)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 2:4])
    ph = normalize_stage_matrix(feat["phenotype_values"])
    x = np.arange(3)
    for i, name in enumerate(feat["phenotype_names"]):
        ax.plot(x, ph[i], marker="o", linewidth=1.8, label=name)
    ax.set_xticks(x, ["Early", "Middle", "Late"])
    ax.set_ylabel("Within-variable normalized stage value", fontsize=12)
    ax.set_title("Phenotype growth trajectory", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    ax = fig.add_subplot(gs[1, 4:6])
    env = normalize_stage_matrix(feat["environment_values"])
    for i, name in enumerate(feat["environment_names"]):
        ax.plot(x, env[i], marker="o", linewidth=1.8, label=name)
    ax.set_xticks(x, ["Early", "Middle", "Late"])
    ax.set_ylabel("Within-variable normalized stage value", fontsize=12)
    ax.set_title("Greenhouse environmental trajectory", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    ax = fig.add_subplot(gs[2, 0:3])
    for name, sig in feat["signatures"].items():
        z = np.asarray(sig, float)
        z = z / max(np.linalg.norm(z), 1e-8)
        ax.plot(np.arange(len(z)), z, marker="o", linewidth=1.8, label=name)
    ax.set_xticks(np.arange(len(SIGNATURE_NAMES)), SIGNATURE_NAMES, rotation=35, ha="right")
    ax.set_ylabel("L2-normalized trajectory signature", fontsize=12)
    ax.set_title("Modality-specific growth signatures", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)

    ax = fig.add_subplot(gs[2, 3:6])
    names = ["Image", "Phenotype", "Environment"]
    mat = np.eye(3, dtype=float)
    for i in range(3):
        for j in range(i + 1, 3):
            mat[i, j] = mat[j, i] = cosine(feat["signatures"][names[i]], feat["signatures"][names[j]])
    im = ax.imshow(mat, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(3), names)
    ax.set_yticks(np.arange(3), names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_title("Cross-Modal Growth Concordance\n(pairwise trajectory cosine)", fontsize=15, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"CMGT-DINOv3 multimodal processing example – {group} (year {year}, final yield {y:.1f})",
        fontsize=19, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.015,
        "RGB uses cached frozen DINOv3 embeddings. Phenotype and environmental curves are normalized only for visualization; "
        "the prediction pipeline uses fold-wise processed numeric features.",
        ha="center", fontsize=11,
    )
    save_figure(fig, out_dir / "Figure_CMGT_DINOv3_Multimodal_Processing.png", dpi)
    return feat


def make_concordance_figure(group, manifest, feat, out_dir, dpi):
    g = manifest[manifest["group"] == group]
    y = float(g["target"].median())
    names = ["Image", "Phenotype", "Environment"]
    mat = np.eye(3, dtype=float)
    for i in range(3):
        for j in range(i + 1, 3):
            mat[i, j] = mat[j, i] = cosine(feat["signatures"][names[i]], feat["signatures"][names[j]])
    vals = [mat[0, 1], mat[0, 2], mat[1, 2]]
    tri_mean, tri_std = float(np.mean(vals)), float(np.std(vals))

    fig = plt.figure(figsize=(14, 6.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.6], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(mat, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(3), names, rotation=25, ha="right")
    ax.set_yticks(np.arange(3), names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_title("Pairwise trajectory concordance", fontsize=16, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 1])
    xpos = np.arange(len(SIGNATURE_NAMES))
    width = 0.24
    for k, name in enumerate(names):
        sig = np.asarray(feat["signatures"][name], float)
        sig = sig / max(np.linalg.norm(sig), 1e-8)
        ax.bar(xpos + (k - 1) * width, sig, width=width, label=name)
    ax.set_xticks(xpos, SIGNATURE_NAMES, rotation=35, ha="right")
    ax.set_ylabel("L2-normalized signature value", fontsize=12)
    ax.set_title("Growth-signature components used for concordance", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Cross-modal concordance example – {group} | yield={y:.1f} | "
        f"tri-modal cosine mean={tri_mean:.3f}, SD={tri_std:.3f}",
        fontsize=18, fontweight="bold",
    )
    save_figure(fig, out_dir / "Figure_CMGT_DINOv3_CrossModal_Concordance.png", dpi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default=None, help="Exact plant-year group, e.g. 2024_CK11")
    parser.add_argument("--year", type=int, default=2024, help="Year for automatic representative sample selection")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    manifest = safe_read_manifest(cfg)
    backbone, emb, emb_meta, p2i = load_embedding_cache(cfg)

    print("[INFO] Proposed backbone:", backbone)
    print("[INFO] Embedding matrix:", emb.shape)
    print("[INFO] Manifest plant-year groups:", manifest["group"].nunique())

    basename_index = build_basename_index(cfg["data"]["root"])
    sensor_repo = SensorRepository(cfg["data"]["root"], cfg["data"]["sensor_features"])

    out_dir = Path(cfg["data"]["output_dir"]) / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_three_year_dataset_montage(manifest, cfg, basename_index, out_dir, args.dpi)

    if args.group:
        group = args.group
        if group not in set(manifest["group"].astype(str)):
            raise ValueError(f"Group '{group}' is not present in the manifest")
    else:
        group = representative_group(manifest, cfg, basename_index, year=args.year)

    print("[INFO] Representative sample:", group)

    feat = make_multimodal_processing_figure(
        group, manifest, cfg, emb, p2i, basename_index, sensor_repo, out_dir, args.dpi
    )
    make_concordance_figure(group, manifest, feat, out_dir, args.dpi)

    print("\n==============================================")
    print("CMGT-DINOv3 paper figures completed.")
    print("Representative group:", group)
    print("Output directory:", out_dir)
    print("==============================================")


if __name__ == "__main__":
    main()
