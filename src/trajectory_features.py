from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .image_paths import split_photo_raw, resolve_image, build_basename_index
from .utils import linear_slope, cosine, safe_corr


STAGES = ("early", "middle", "late")


def _split_stages(g):
    g = g.sort_values("date", na_position="last").copy()
    n = len(g)

    # Chronological thirds by observation count. Robust to irregular acquisition dates.
    a = max(1, int(np.ceil(n / 3)))
    b = max(a + 1, int(np.ceil(2 * n / 3)))

    return {
        "early": g.iloc[:a],
        "middle": g.iloc[a:b] if b > a else g.iloc[:a],
        "late": g.iloc[b:] if b < n else g.iloc[-a:],
    }


def _nan_summary(a):
    a = np.asarray(a, float)
    ok = a[np.isfinite(a)]
    if not len(ok):
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "last": np.nan,
            "missing": 1.0,
        }
    return {
        "mean": float(np.mean(ok)),
        "std": float(np.std(ok)),
        "min": float(np.min(ok)),
        "max": float(np.max(ok)),
        "q25": float(np.quantile(ok, 0.25)),
        "q75": float(np.quantile(ok, 0.75)),
        "last": float(ok[-1]),
        "missing": float(1.0 - len(ok) / max(len(a), 1)),
    }


def _relative_delta(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    den = np.abs(a) + np.abs(b) + 1e-6
    return 2.0 * (b - a) / den


def _trajectory_signature(e, m, l, early_var=0.0, late_var=0.0, normalize_vectors=True):
    e = np.nan_to_num(np.asarray(e, float), nan=0.0)
    m = np.nan_to_num(np.asarray(m, float), nan=0.0)
    l = np.nan_to_num(np.asarray(l, float), nan=0.0)

    if normalize_vectors:
        def unit(x):
            n = np.linalg.norm(x)
            return x / max(n, 1e-8)
        ee, mm, ll = unit(e), unit(m), unit(l)
    else:
        scale = np.abs(np.concatenate([e, m, l]))
        scale = np.nanmedian(scale[scale > 0]) if np.any(scale > 0) else 1.0
        ee, mm, ll = e / scale, m / scale, l / scale

    dem = mm - ee
    dml = ll - mm
    del_ = ll - ee

    em = float(np.linalg.norm(dem))
    ml = float(np.linalg.norm(dml))
    el = float(np.linalg.norm(del_))
    accel = ml - em
    direction = cosine(dem, dml)

    return np.asarray(
        [em, ml, el, accel, direction, float(early_var), float(late_var)],
        dtype=np.float32,
    )


def _pair_concordance(name, a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    # Dimensionless normalization for cross-modality comparison.
    aa = a / max(np.linalg.norm(a), 1e-8)
    bb = b / max(np.linalg.norm(b), 1e-8)

    vals = {
        f"{name}__cosine": cosine(aa, bb),
        f"{name}__correlation": safe_corr(aa, bb),
        f"{name}__mean_abs_difference": float(np.mean(np.abs(aa - bb))),
        f"{name}__em_product": float(a[0] * b[0]),
        f"{name}__ml_product": float(a[1] * b[1]),
        f"{name}__el_product": float(a[2] * b[2]),
        f"{name}__acceleration_agreement": float(1.0 / (1.0 + abs(a[3] - b[3]))),
        f"{name}__direction_product": float(a[4] * b[4]),
    }
    return vals


def _vpd_kpa(temp_c, rh):
    t = np.asarray(temp_c, float)
    h = np.asarray(rh, float)
    es = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
    return es * (1.0 - np.clip(h, 0.0, 100.0) / 100.0)


def _sensor_frame_with_derived(frame):
    x = frame.copy()

    if "Air Temperature" in x.columns and "Relative Humidity" in x.columns:
        x["VPD"] = _vpd_kpa(
            pd.to_numeric(x["Air Temperature"], errors="coerce"),
            pd.to_numeric(x["Relative Humidity"], errors="coerce"),
        )
    else:
        x["VPD"] = np.nan

    return x


def _sensor_stage_features(frames, cfg):
    features = list(cfg["data"]["sensor_features"]) + ["VPD"]
    interval = float(cfg["trajectory"].get("sensor_interval_hours", 0.5))
    base = float(cfg["trajectory"].get("growing_degree_base_c", 10.0))
    high_t = float(cfg["trajectory"].get("high_temperature_c", 30.0))

    if frames:
        x = pd.concat([_sensor_frame_with_derived(f) for f in frames], ignore_index=True)
    else:
        x = pd.DataFrame(columns=features)

    out = {}
    means = []
    variability = []

    for f in features:
        a = pd.to_numeric(x.get(f, pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        s = _nan_summary(a)
        means.append(s["mean"])
        variability.append(s["std"])
        for k in ["mean", "std", "min", "max", "q25", "q75"]:
            out[f"{f}__{k}"] = s[k]

    t = pd.to_numeric(x.get("Air Temperature", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    light = pd.to_numeric(x.get("Light Intensity", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    co2 = pd.to_numeric(x.get("CO2", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    vpd = pd.to_numeric(x.get("VPD", pd.Series(dtype=float)), errors="coerce").to_numpy(float)

    out["growing_degree_hours"] = float(np.nansum(np.maximum(t - base, 0.0)) * interval) if len(t) else np.nan
    out["high_temperature_hours"] = float(np.nansum(t > high_t) * interval) if len(t) else np.nan
    out["vpd_exposure"] = float(np.nansum(np.maximum(vpd, 0.0)) * interval) if len(vpd) else np.nan
    out["light_exposure_proxy"] = float(np.nansum(np.maximum(light, 0.0)) * interval) if len(light) else np.nan
    out["co2_exposure_proxy"] = float(np.nansum(np.maximum(co2, 0.0)) * interval) if len(co2) else np.nan
    out["observed_fraction"] = float(np.isfinite(x.to_numpy(float)).mean()) if len(x) else 0.0

    return out, np.asarray(means, np.float32), float(np.nanmean(variability)) if len(variability) else 0.0


def _image_stage_features(stage_frame, emb, p2i, dataset_root, basename_index):
    paths = []

    for r in stage_frame.itertuples():
        for raw in split_photo_raw(getattr(r, "photo_raw", None)):
            p = resolve_image(
                raw,
                dataset_root,
                getattr(r, "manual_csv", None),
                basename_index,
            )
            if p and p in p2i:
                paths.append(p)

    paths = list(dict.fromkeys(paths))

    if not paths:
        d = emb.shape[1]
        return (
            np.full(d, np.nan, np.float32),
            np.full(d, np.nan, np.float32),
            0,
        )

    z = np.stack([emb[p2i[p]] for p in paths]).astype(np.float32)
    return z.mean(0), z.std(0), len(paths)


def _phenotype_stage_features(stage_frame, tab_cols):
    out = {}
    means = []
    vars_ = []

    for c in tab_cols:
        a = pd.to_numeric(stage_frame[c], errors="coerce").to_numpy(float)
        s = _nan_summary(a)
        means.append(s["mean"])
        vars_.append(s["std"])

        for k in ["mean", "std", "last", "missing"]:
            out[f"{c}__{k}"] = s[k]

    return out, np.asarray(means, np.float32), float(np.nanmean(vars_)) if len(vars_) else 0.0


def build_cmgt_samples(manifest, embeddings, embedding_meta, sensor_repo, cfg):
    manifest = manifest.copy()
    manifest["date"] = pd.to_datetime(manifest["date"], errors="coerce")

    if "group" not in manifest.columns:
        manifest["group"] = (
            manifest["year"].astype(str)
            + "_"
            + manifest["plant_id"].astype(str)
        )

    tab_cols = [
        c for c in manifest.columns
        if c.startswith("tab__")
        and pd.to_numeric(manifest[c], errors="coerce").notna().any()
    ]

    print(f"[FEATURE] Longitudinal phenotype variables: {len(tab_cols)}")

    p2i = dict(
        zip(
            embedding_meta["image_path"].astype(str),
            embedding_meta["index"].astype(int),
        )
    )

    basename_index = build_basename_index(cfg["data"]["root"])

    blocks = {
        "image_trajectory": [],
        "phenotype_trajectory": [],
        "environment_trajectory": [],
        "concordance": [],
        "meta_reliability": [],
        "static_image": [],
        "static_phenotype": [],
        "static_environment": [],
    }

    names = {k: None for k in blocks}
    meta_rows = []
    targets = []

    for group, g0 in manifest.groupby("group", sort=True):
        g0 = g0.sort_values("date", na_position="last")

        y = pd.to_numeric(g0["target"], errors="coerce").dropna()
        if not len(y):
            continue
        target = float(y.median())

        stages = _split_stages(g0)

        # ----------------------------------------------------------
        # IMAGE TRAJECTORY
        # ----------------------------------------------------------
        img_mean = {}
        img_std = {}
        img_count = {}

        for stage in STAGES:
            mu, sd, count = _image_stage_features(
                stages[stage],
                embeddings,
                p2i,
                cfg["data"]["root"],
                basename_index,
            )
            img_mean[stage] = mu
            img_std[stage] = sd
            img_count[stage] = count

        img_delta_em = img_mean["middle"] - img_mean["early"]
        img_delta_ml = img_mean["late"] - img_mean["middle"]
        img_delta_el = img_mean["late"] - img_mean["early"]

        image_vector = np.concatenate([
            img_mean["early"],
            img_mean["middle"],
            img_mean["late"],
            img_delta_em,
            img_delta_ml,
            img_delta_el,
        ]).astype(np.float32)

        image_sig = _trajectory_signature(
            img_mean["early"],
            img_mean["middle"],
            img_mean["late"],
            early_var=float(np.nanmean(img_std["early"])),
            late_var=float(np.nanmean(img_std["late"])),
            normalize_vectors=True,
        )

        static_image = np.concatenate([
            np.nanmean(np.stack(list(img_mean.values())), axis=0),
            np.nanmean(np.stack(list(img_std.values())), axis=0),
        ]).astype(np.float32)

        # ----------------------------------------------------------
        # PHENOTYPE TRAJECTORY
        # ----------------------------------------------------------
        ph_stage_dict = {}
        ph_means = {}
        ph_vars = {}

        for stage in STAGES:
            d, means, var = _phenotype_stage_features(stages[stage], tab_cols)
            ph_stage_dict[stage] = d
            ph_means[stage] = means
            ph_vars[stage] = var

        ph_values = {}
        for c_idx, c in enumerate(tab_cols):
            for stage in STAGES:
                for stat in ["mean", "std", "last", "missing"]:
                    ph_values[f"{c}__{stage}__{stat}"] = ph_stage_dict[stage].get(
                        f"{c}__{stat}", np.nan
                    )

            e = ph_means["early"][c_idx]
            m = ph_means["middle"][c_idx]
            l = ph_means["late"][c_idx]

            ph_values[f"{c}__delta_em"] = m - e
            ph_values[f"{c}__delta_ml"] = l - m
            ph_values[f"{c}__delta_el"] = l - e
            ph_values[f"{c}__relative_delta_el"] = (
                2.0 * (l - e) / (abs(l) + abs(e) + 1e-6)
                if np.isfinite(l) and np.isfinite(e) else np.nan
            )

            allv = pd.to_numeric(g0[c], errors="coerce").to_numpy(float)
            ph_values[f"{c}__global_slope"] = linear_slope(allv)
            ph_values[f"{c}__global_missing"] = float(
                1.0 - np.isfinite(allv).sum() / max(len(allv), 1)
            )

        phenotype_vector_names = list(ph_values)
        phenotype_vector = np.asarray([ph_values[k] for k in phenotype_vector_names], np.float32)

        # Group-relative normalization before signature calculation.
        ph_scale = np.nanmean(np.abs(np.vstack([
            ph_means["early"],
            ph_means["middle"],
            ph_means["late"],
        ])), axis=0) + np.nanstd(np.vstack([
            ph_means["early"],
            ph_means["middle"],
            ph_means["late"],
        ]), axis=0) + 1e-6

        pe = ph_means["early"] / ph_scale
        pm = ph_means["middle"] / ph_scale
        pl = ph_means["late"] / ph_scale

        ph_sig = _trajectory_signature(
            pe, pm, pl,
            early_var=ph_vars["early"],
            late_var=ph_vars["late"],
            normalize_vectors=False,
        )

        static_ph_values = {}
        for c in tab_cols:
            a = pd.to_numeric(g0[c], errors="coerce").to_numpy(float)
            s = _nan_summary(a)
            for stat in ["mean", "std", "last", "missing"]:
                static_ph_values[f"{c}__{stat}"] = s[stat]
            static_ph_values[f"{c}__slope"] = linear_slope(a)

        static_phenotype_names = list(static_ph_values)
        static_phenotype = np.asarray(
            [static_ph_values[k] for k in static_phenotype_names],
            np.float32,
        )

        # ----------------------------------------------------------
        # ENVIRONMENT TRAJECTORY
        # ----------------------------------------------------------
        env_stage_dict = {}
        env_means = {}
        env_vars = {}
        sensor_modes = []
        direct_sensor_files = 0

        for stage in STAGES:
            frames, mode, direct_count = sensor_repo.frames_for_rows(stages[stage])
            d, means, var = _sensor_stage_features(frames, cfg)
            env_stage_dict[stage] = d
            env_means[stage] = means
            env_vars[stage] = var
            sensor_modes.append(mode)
            direct_sensor_files += direct_count

        env_values = {}
        env_base_names = list(env_stage_dict["early"].keys())

        for stage in STAGES:
            for k, v in env_stage_dict[stage].items():
                env_values[f"{stage}__{k}"] = v

        sensor_mean_names = list(cfg["data"]["sensor_features"]) + ["VPD"]
        for j, name in enumerate(sensor_mean_names):
            e = env_means["early"][j]
            m = env_means["middle"][j]
            l = env_means["late"][j]
            env_values[f"{name}__delta_em"] = m - e
            env_values[f"{name}__delta_ml"] = l - m
            env_values[f"{name}__delta_el"] = l - e

        environment_vector_names = list(env_values)
        environment_vector = np.asarray(
            [env_values[k] for k in environment_vector_names],
            np.float32,
        )

        env_scale = np.nanmean(np.abs(np.vstack([
            env_means["early"],
            env_means["middle"],
            env_means["late"],
        ])), axis=0) + np.nanstd(np.vstack([
            env_means["early"],
            env_means["middle"],
            env_means["late"],
        ]), axis=0) + 1e-6

        ee = env_means["early"] / env_scale
        em = env_means["middle"] / env_scale
        el = env_means["late"] / env_scale

        env_sig = _trajectory_signature(
            ee, em, el,
            early_var=env_vars["early"],
            late_var=env_vars["late"],
            normalize_vectors=False,
        )

        # Static environment = whole-season stage-average summaries.
        static_env_values = {}
        for key in env_base_names:
            vals = [env_stage_dict[s].get(key, np.nan) for s in STAGES]
            static_env_values[f"{key}__stage_mean"] = float(np.nanmean(vals))
            static_env_values[f"{key}__stage_std"] = float(np.nanstd(vals))

        static_environment_names = list(static_env_values)
        static_environment = np.asarray(
            [static_env_values[k] for k in static_environment_names],
            np.float32,
        )

        # ----------------------------------------------------------
        # CROSS-MODAL GROWTH CONCORDANCE
        # ----------------------------------------------------------
        concordance = {}
        concordance.update(_pair_concordance("image_phenotype", image_sig, ph_sig))
        concordance.update(_pair_concordance("image_environment", image_sig, env_sig))
        concordance.update(_pair_concordance("phenotype_environment", ph_sig, env_sig))

        pair_cosines = [
            concordance["image_phenotype__cosine"],
            concordance["image_environment__cosine"],
            concordance["phenotype_environment__cosine"],
        ]

        concordance["tri_modal_cosine_mean"] = float(np.mean(pair_cosines))
        concordance["tri_modal_cosine_std"] = float(np.std(pair_cosines))
        concordance["tri_modal_growth_product"] = float(
            image_sig[2] * ph_sig[2] * env_sig[2]
        )

        # Add trajectory signatures explicitly so the model can distinguish
        # concordance from each modality's own dynamics.
        for prefix, sig in [
            ("image_signature", image_sig),
            ("phenotype_signature", ph_sig),
            ("environment_signature", env_sig),
        ]:
            for j, name in enumerate([
                "em_magnitude", "ml_magnitude", "el_magnitude",
                "acceleration", "direction_consistency",
                "early_variability", "late_variability",
            ]):
                concordance[f"{prefix}__{name}"] = float(sig[j])

        concordance_names = list(concordance)
        concordance_vector = np.asarray(
            [concordance[k] for k in concordance_names],
            np.float32,
        )

        # ----------------------------------------------------------
        # RELIABILITY / ACQUISITION META
        # ----------------------------------------------------------
        date_series = pd.to_datetime(g0["date"], errors="coerce").dropna()
        duration_days = (
            int((date_series.max() - date_series.min()).days)
            if len(date_series) >= 2 else 0
        )

        ph_obs = float(
            np.isfinite(
                g0[tab_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            ).mean()
        ) if tab_cols else 0.0

        meta_values = {
            "image_count_early": img_count["early"],
            "image_count_middle": img_count["middle"],
            "image_count_late": img_count["late"],
            "image_count_total": sum(img_count.values()),
            "phenotype_observed_fraction": ph_obs,
            "direct_sensor_files": direct_sensor_files,
            "sensor_used_direct_any": float("direct" in sensor_modes),
            "duration_days": duration_days,
            "longitudinal_rows": len(g0),
        }

        meta_names = list(meta_values)
        meta_vector = np.asarray([meta_values[k] for k in meta_names], np.float32)

        # ----------------------------------------------------------
        # APPEND
        # ----------------------------------------------------------
        blocks["image_trajectory"].append(image_vector)
        blocks["phenotype_trajectory"].append(phenotype_vector)
        blocks["environment_trajectory"].append(environment_vector)
        blocks["concordance"].append(concordance_vector)
        blocks["meta_reliability"].append(meta_vector)

        blocks["static_image"].append(static_image)
        blocks["static_phenotype"].append(static_phenotype)
        blocks["static_environment"].append(static_environment)

        if names["image_trajectory"] is None:
            d = embeddings.shape[1]
            names["image_trajectory"] = (
                [f"image_early_{i}" for i in range(d)]
                + [f"image_middle_{i}" for i in range(d)]
                + [f"image_late_{i}" for i in range(d)]
                + [f"image_delta_em_{i}" for i in range(d)]
                + [f"image_delta_ml_{i}" for i in range(d)]
                + [f"image_delta_el_{i}" for i in range(d)]
            )
            names["phenotype_trajectory"] = phenotype_vector_names
            names["environment_trajectory"] = environment_vector_names
            names["concordance"] = concordance_names
            names["meta_reliability"] = meta_names
            names["static_image"] = (
                [f"static_image_mean_{i}" for i in range(d)]
                + [f"static_image_std_{i}" for i in range(d)]
            )
            names["static_phenotype"] = static_phenotype_names
            names["static_environment"] = static_environment_names

        meta_rows.append({
            "group": group,
            "year": int(g0["year"].iloc[0]),
            "plant_id": str(g0["plant_id"].iloc[0]),
            "target": target,
            "rows": len(g0),
            "duration_days": duration_days,
            "image_count_total": sum(img_count.values()),
            "sensor_mode": "direct" if "direct" in sensor_modes else "date_fallback",
            "direct_sensor_files": direct_sensor_files,
        })
        targets.append(target)

    if not targets:
        raise RuntimeError("No CMGT-DINO plant-year samples constructed.")

    data = {
        k: np.stack(v).astype(np.float32)
        for k, v in blocks.items()
    }
    data["feature_names"] = names
    data["meta"] = pd.DataFrame(meta_rows)
    data["y"] = np.asarray(targets, float)

    print(f"[FEATURE] Plant-year samples: {len(targets)}")
    for k in blocks:
        print(f"[FEATURE] {k}: {data[k].shape}")

    return data
