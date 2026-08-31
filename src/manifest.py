from __future__ import annotations
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

try:
    from docx import Document
except Exception:
    Document = None

ID_ALIASES = [
    "Number", "Plant Number", "Plant No", "Plant ID", "PlantID",
    "Order Number", "Sample Number", "编号", "株号", "植株编号",
]
DATE_ALIASES = ["Date", "Measurement Date", "Sampling Date", "日期", "测量日期", "采样日期"]
PHOTO_ALIASES = ["Photo Path", "PhotoPath", "Image Path", "ImagePath", "Photo", "Image", "照片路径", "图片路径", "图像路径"]
SENSOR_PATH_ALIASES = ["Sensor Path", "SensorPath", "Sensor File", "Sensor Data Path", "传感器路径", "传感器文件", "传感数据路径"]
TARGET_ALIASES = ["yield per tree", "yield per plant", "total yield", "plant yield", "yield", "production", "产量", "单株产量", "总产量"]
TARGET_EXCLUDES = ["single fruit weight", "fruit weight", "average fruit weight", "单果重", "fruit diameter", "果径"]
PHENOTYPE_ALIASES = {
    "ndvi": ["NDVI"], "rvi": ["RVI"], "lnc": ["LNC"], "lna": ["LNA"],
    "lai": ["LAI"], "ldw": ["LDW"],
    "plant_height": ["Plant Height", "Height", "株高"],
    "stem_diameter": ["Stem Diameter", "Stem Dia", "茎粗", "茎径"],
    "leaf_width": ["Leaf Width", "叶宽"], "leaf_length": ["Leaf Length", "叶长"],
    "leaf_area": ["Leaf Area", "叶面积"], "leaf_angle": ["Leaf Angle", "叶角"],
    "number_of_leaves": ["Number Of Leaves", "Number of Leaves", "Leaf Number", "叶片数", "叶数"],
}


def norm_text(x):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(x).strip().lower())


def normalize_plant_id(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    if re.fullmatch(r"[-+]?\d+\.0+", s):
        s = s.split(".")[0]
    return re.sub(r"\s+", "", s).upper()



def compact_plant_id(x):
    """
    Remove separators while preserving letters.

    Examples
    --------
    1-2-1 -> 121
    CK-2-1 -> CK21
    L111 -> L111
    """
    s = normalize_plant_id(x)
    if not s:
        return None
    return re.sub(r"[^A-Z0-9]+", "", s)


def numeric_core(x):
    """
    Digits-only plant-position code.

    The uploaded Horti-M3 ID audit shows:
      2023: 1-2-1 -> growth ID 121
      2025: 1-1-1 -> growth ID with numeric core 111
    """
    s = normalize_plant_id(x)
    if not s:
        return None
    d = "".join(re.findall(r"\d+", s))
    if not d:
        return None
    return d.lstrip("0") or "0"


def id_variants(x):
    """
    Backward-compatible variants used for diagnostics.
    """
    s = normalize_plant_id(x)
    if not s:
        return set()
    out = {s}

    compact = compact_plant_id(s)
    if compact:
        out.add(compact)

    core = numeric_core(s)
    if core:
        out.add(core)

    m = re.fullmatch(r"[A-Z]+(\d+)", compact or "")
    if m:
        out.add(m.group(1).lstrip("0") or "0")

    return out


def _unique_lookup(values, key_fn):
    """
    Build key -> original ID only for keys that are unique.

    Ambiguous mappings are deliberately excluded rather than guessed.
    """
    buckets = {}
    for value in values:
        key = key_fn(value)
        if not key:
            continue
        buckets.setdefault(key, []).append(value)

    return {
        key: vals[0]
        for key, vals in buckets.items()
        if len(set(vals)) == 1
    }


def build_growth_identity_index(growth_ids):
    growth_ids = [
        normalize_plant_id(x)
        for x in growth_ids
    ]
    growth_ids = [
        x for x in growth_ids
        if x
    ]

    return {
        "exact": _unique_lookup(
            growth_ids,
            lambda x: normalize_plant_id(x),
        ),
        "compact": _unique_lookup(
            growth_ids,
            compact_plant_id,
        ),
        "numeric": _unique_lookup(
            growth_ids,
            numeric_core,
        ),
    }


def harmonize_target_id(target_id, growth_index, year=None):
    """
    Horti-M3 plant-identity harmonization.

    Matching order is intentionally conservative and uses the actual
    2023/2024/2025 ID diagnostic:

    1. exact normalized ID;
    2. separator-insensitive alphanumeric ID;
    3. 2025 position bridge: a-b-c -> Labc when that exact growth ID exists;
    4. 2023 CK bridge: CK-a-b -> 7ab;
    5. unique numeric-position core fallback.

    A mapping is never accepted from an ambiguous numeric core.
    """
    tid = normalize_plant_id(target_id)
    if not tid:
        return None, "unmatched"

    if tid in growth_index["exact"]:
        return growth_index["exact"][tid], "exact"

    compact = compact_plant_id(tid)
    if compact and compact in growth_index["compact"]:
        return growth_index["compact"][compact], "compact_separator"

    core = numeric_core(tid)

    # Uploaded 2025 diagnostic:
    # production IDs such as 1-1-1 correspond to growth IDs such as L111.
    if year is not None and int(year) == 2025:
        if core:
            candidate = "L" + core
            if candidate in growth_index["exact"]:
                return growth_index["exact"][candidate], "year2025_L_position"

    # Uploaded 2023 diagnostic:
    # CK-2-1 ... CK-2-6 correspond to growth IDs 721 ... 726.
    if tid.startswith("CK") and core:
        ck_growth_core = "7" + core
        if ck_growth_core in growth_index["numeric"]:
            return growth_index["numeric"][ck_growth_core], "ck_to_7prefix"

    if core and core in growth_index["numeric"]:
        return growth_index["numeric"][core], "numeric_core_unique"

    return None, "unmatched"

def match_header(columns, aliases):
    cols = list(columns)
    nmap = {norm_text(c): c for c in cols}
    for a in aliases:
        na = norm_text(a)
        if na in nmap:
            return nmap[na]
    for c in cols:
        nc = norm_text(c)
        for a in aliases:
            na = norm_text(a)
            if na and (na in nc or nc in na):
                return c
    return None


def read_csv_flexible(path):
    last = None
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last = e
    raise RuntimeError(f"Cannot read {path}: {last}")


def read_tables(path):
    path = Path(path)
    ext = path.suffix.lower()
    out = []
    if ext == ".csv":
        try:
            out.append(("csv", read_csv_flexible(path)))
        except Exception:
            pass
        return out
    if ext in {".xlsx", ".xls"}:
        try:
            for name, df in pd.read_excel(path, sheet_name=None).items():
                if df is not None and not df.empty:
                    out.append((str(name), df))
        except Exception:
            pass
        return out
    if ext == ".docx" and Document is not None:
        try:
            doc = Document(path)
            for ti, table in enumerate(doc.tables):
                rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                if len(rows) < 2:
                    continue
                width = max(len(r) for r in rows)
                rows = [r + [""] * (width - len(r)) for r in rows]
                seen, cols = {}, []
                for j, h in enumerate(rows[0]):
                    h = h.strip() or f"col_{j}"
                    n = seen.get(h, 0)
                    seen[h] = n + 1
                    cols.append(h if n == 0 else f"{h}_{n}")
                df = pd.DataFrame(rows[1:], columns=cols)
                if not df.empty:
                    out.append((f"table_{ti}", df))
        except Exception:
            pass
    return out


def infer_date_from_filename(path, year=None):
    s = Path(path).stem
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", s)
    if m:
        try:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass
    m = re.search(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})", s)
    if m:
        try:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass
    return pd.NaT



def find_date_column(df):
    """
    Detect a true date column without attempting to parse every numeric
    phenotype column. This removes the pandas date-inference warning seen
    in the previous run and avoids false date detection.
    """
    c = match_header(df.columns, DATE_ALIASES)
    if c:
        return c

    date_pattern = re.compile(
        r"20\d{2}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}"
    )

    for c in df.columns:
        s = df[c]
        # Fallback date discovery is limited to textual columns.
        if not (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
        ):
            continue

        sample = s.dropna().astype(str).str.strip()
        if sample.empty:
            continue

        sample = sample.head(100)
        pattern_rate = sample.str.contains(
            date_pattern,
            regex=True,
        ).mean()

        if pattern_rate < 0.50:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                q = pd.to_datetime(
                    sample,
                    errors="coerce",
                    format="mixed",
                )
            except TypeError:
                q = pd.to_datetime(
                    sample,
                    errors="coerce",
                )

        if q.notna().mean() >= 0.70:
            yrs = q.dropna().dt.year
            if (
                len(yrs)
                and yrs.between(2020, 2030).mean() >= 0.90
            ):
                return c

    return None

def path_belongs_to_year(path, root, year):
    try:
        rel = Path(path).relative_to(Path(root))
    except Exception:
        rel = Path(path)
    y = str(year)
    # Horti-M3 uses components such as 2024/2024/202409 and
    # top-level names such as "2024 Tomato Production Data.xlsx".
    return any(str(part) == y or str(part).startswith(y) for part in rel.parts)


def discover_growth_files(root, year):
    root = Path(root)
    candidates = []
    # Files inside year directories.
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".csv", ".xlsx", ".xls", ".docx"}:
            continue
        if not path_belongs_to_year(p, root, year):
            continue
        low = str(p).lower()
        name = p.name.lower()
        if "sensor" in low or "传感" in low or "production" in name or "yield" in name or "产量" in name:
            continue
        if any(k in low for k in ["grow", "growth", "growth_index", "grow_index", "生长"]):
            candidates.append(p)
    uniq = list(dict.fromkeys(candidates))
    uniq.sort(key=lambda p: (0 if p.suffix.lower() == ".csv" else 1, 0 if re.search(r"20\d{6}", p.stem) else 1, len(str(p)), str(p)))
    return uniq


def phenotype_map(df):
    return {k: c for k, aliases in PHENOTYPE_ALIASES.items() if (c := match_header(df.columns, aliases)) is not None}


def parse_growth_table(df, source_path, source_name, year):
    if df is None or df.empty:
        return pd.DataFrame()
    id_col = match_header(df.columns, ID_ALIASES)
    if id_col is None:
        return pd.DataFrame()
    date_col = find_date_column(df)
    file_date = infer_date_from_filename(source_path, year)
    photo_col = match_header(df.columns, PHOTO_ALIASES)
    sensor_col = match_header(df.columns, SENSOR_PATH_ALIASES)
    pheno = phenotype_map(df)
    rows = []
    for _, r in df.iterrows():
        pid = normalize_plant_id(r.get(id_col))
        if not pid:
            continue
        d = pd.NaT
        if date_col is not None:
            q = pd.to_datetime(r.get(date_col), errors="coerce")
            if not pd.isna(q):
                d = pd.Timestamp(q).normalize()
        if pd.isna(d):
            d = file_date
        row = {
            "year": int(year), "date": d, "plant_id": pid, "plant_id_raw": str(r.get(id_col)),
            "manual_csv": str(source_path), "growth_source": str(source_name),
            "photo_raw": str(r.get(photo_col)) if photo_col and pd.notna(r.get(photo_col)) else "",
            "sensor_raw": str(r.get(sensor_col)) if sensor_col and pd.notna(r.get(sensor_col)) else "",
        }
        for canonical, c in pheno.items():
            row[f"tab__{canonical}"] = pd.to_numeric(r.get(c), errors="coerce")
        rows.append(row)
    return pd.DataFrame(rows)


def extract_growth_rows(root, year):
    parts, audit = [], []
    for p in discover_growth_files(root, year):
        for sheet, df in read_tables(p):
            rec = parse_growth_table(df, p, sheet, year)
            audit.append({
                "year": year, "file": str(p), "sheet": sheet, "rows_read": len(df),
                "records_extracted": len(rec), "id_column_detected": match_header(df.columns, ID_ALIASES),
                "date_column_detected": find_date_column(df), "photo_column_detected": match_header(df.columns, PHOTO_ALIASES),
                "sensor_column_detected": match_header(df.columns, SENSOR_PATH_ALIASES),
                "columns": " | ".join(map(str, df.columns)),
            })
            if len(rec):
                parts.append(rec)
    if not parts:
        return pd.DataFrame(), pd.DataFrame(audit)
    out = pd.concat(parts, ignore_index=True, sort=False)
    # Prefer rows containing direct photo/sensor links when duplicates exist.
    out["_richness"] = out["photo_raw"].fillna("").astype(str).str.len().gt(0).astype(int) + out["sensor_raw"].fillna("").astype(str).str.len().gt(0).astype(int)
    out = out.sort_values("_richness", ascending=False)
    # Same plant/date from a daily CSV should not be duplicated by top-level
    # summary tables. Undated 2023 rows are preserved unless exact duplicates,
    # because a single seasonal CSV can contain repeated longitudinal records.
    dated = out[out["date"].notna()].drop_duplicates(subset=["year", "date", "plant_id"], keep="first")
    undated = out[out["date"].isna()].drop_duplicates(keep="first")
    out = pd.concat([dated, undated], ignore_index=True, sort=False).drop(columns=["_richness"])
    return out.reset_index(drop=True), pd.DataFrame(audit)


def discover_target_files(root, year):
    root = Path(root)
    out = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".csv", ".xlsx", ".xls", ".docx"}:
            continue
        if not path_belongs_to_year(p, root, year):
            continue
        low = p.name.lower()
        if "production" in low or "yield" in low or "产量" in low:
            out.append(p)
    return sorted(set(out), key=str)


def target_col_score(c):
    nc = norm_text(c)
    if any(norm_text(x) in nc for x in TARGET_EXCLUDES):
        return -1000
    score = 0
    for alias in TARGET_ALIASES:
        na = norm_text(alias)
        if nc == na:
            score = max(score, 100)
        elif na in nc:
            score = max(score, 80)
    if "yieldpertree" in nc or "yieldperplant" in nc:
        score = max(score, 120)
    if "totalyield" in nc or "总产量" in str(c):
        score = max(score, 115)
    return score


def id_col_score(c):
    nc = norm_text(c)
    score = 0
    for rank, alias in enumerate(ID_ALIASES):
        na = norm_text(alias)
        if nc == na:
            score = max(score, 100 - rank)
        elif na in nc:
            score = max(score, 70 - rank)
    return score


def numeric_series(s):
    q = s.astype(str).str.replace(",", "", regex=False).str.replace("g", "", regex=False).str.replace("G", "", regex=False).str.strip()
    return pd.to_numeric(q, errors="coerce")



def candidate_target_table(
    df,
    source_path,
    sheet,
    year,
    growth_ids,
    override=None,
):
    override = override or {}

    if df is None or df.empty:
        return None, None

    if override.get("id_column") in df.columns:
        id_col = override["id_column"]
    else:
        ranked = sorted(
            [(id_col_score(c), c) for c in df.columns],
            reverse=True,
        )
        id_col = (
            ranked[0][1]
            if ranked and ranked[0][0] > 0
            else None
        )

    if override.get("yield_column") in df.columns:
        y_col = override["yield_column"]
        yscore = 1000
    else:
        ranked = sorted(
            [(target_col_score(c), c) for c in df.columns],
            reverse=True,
        )
        yscore, y_col = (
            ranked[0]
            if ranked
            else (0, None)
        )
        if yscore <= 0:
            y_col = None

    audit = {
        "year": year,
        "file": str(source_path),
        "sheet": str(sheet),
        "id_col": id_col,
        "yield_col": y_col,
        "yield_col_score": yscore,
        "rows": len(df),
        "columns": " | ".join(map(str, df.columns)),
        "growth_unique_ids": int(len(set(growth_ids))),
        "unique_target_ids": 0,
        "matched_target_ids": 0,
        "unmatched_target_ids": 0,
        "target_id_match_rate": 0.0,
        "growth_target_coverage": 0.0,
        "nonmissing_yield": 0,
        "mapping_methods": "",
    }

    if id_col is None or y_col is None:
        return None, audit

    tmp = pd.DataFrame(
        {
            "plant_id_raw": df[id_col].astype(str),
            "plant_id": df[id_col].map(normalize_plant_id),
            "target": numeric_series(df[y_col]),
        }
    ).dropna(
        subset=["plant_id", "target"]
    )

    if tmp.empty:
        return None, audit

    # One final yield row per plant remains unchanged. If the production
    # source contains several harvest-yield rows per plant, the values are
    # summed into the final season yield.
    tgt = (
        tmp.groupby(
            "plant_id",
            as_index=False,
        )
        .agg(
            target=("target", "sum"),
            target_rows=("target", "size"),
            plant_id_raw=("plant_id_raw", "first"),
        )
    )

    growth_index = build_growth_identity_index(
        growth_ids
    )

    mapped = []
    methods = []

    for pid in tgt["plant_id"]:
        gid, method = harmonize_target_id(
            pid,
            growth_index,
            year=year,
        )
        mapped.append(gid)
        methods.append(method)

    tgt["growth_plant_id"] = mapped
    tgt["mapping_method"] = methods

    matched_mask = tgt["growth_plant_id"].notna()

    audit["unique_target_ids"] = int(
        tgt["plant_id"].nunique()
    )
    audit["nonmissing_yield"] = int(
        len(tgt)
    )
    audit["matched_target_ids"] = int(
        matched_mask.sum()
    )
    audit["unmatched_target_ids"] = int(
        (~matched_mask).sum()
    )
    audit["target_id_match_rate"] = float(
        matched_mask.sum()
        / max(len(tgt), 1)
    )
    audit["growth_target_coverage"] = float(
        tgt.loc[
            matched_mask,
            "growth_plant_id",
        ].nunique()
        / max(len(set(growth_ids)), 1)
    )

    method_counts = (
        tgt.loc[
            matched_mask,
            "mapping_method",
        ]
        .value_counts()
        .to_dict()
    )
    audit["mapping_methods"] = ";".join(
        f"{k}:{v}"
        for k, v in sorted(method_counts.items())
    )

    return tgt, audit


def extract_targets(
    root,
    year,
    growth_ids,
    override=None,
):
    override = override or {}
    candidates = []
    audits = []

    for p in discover_target_files(
        root,
        year,
    ):
        if (
            override.get("file_contains")
            and override["file_contains"].lower()
            not in p.name.lower()
        ):
            continue

        for sheet, df in read_tables(p):
            if (
                override.get("sheet_contains")
                and override["sheet_contains"].lower()
                not in str(sheet).lower()
            ):
                continue

            tgt, audit = candidate_target_table(
                df,
                p,
                sheet,
                year,
                growth_ids,
                override,
            )

            if audit is not None:
                audits.append(audit)

            if tgt is not None:
                # Prefer complete target-ID alignment. Coverage of all growth
                # plants is informational because 2023 and 2025 production
                # files legitimately contain only subsets of the growth IDs.
                score = (
                    audit["target_id_match_rate"] * 1_000_000
                    + audit["matched_target_ids"] * 1_000
                    + max(audit["yield_col_score"], 0)
                )
                candidates.append(
                    (
                        score,
                        p,
                        sheet,
                        tgt,
                        audit,
                    )
                )

    if not candidates:
        return (
            pd.DataFrame(),
            pd.DataFrame(audits),
            None,
        )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    _, p, sheet, tgt, best = candidates[0]

    rows = []

    for r in tgt.itertuples():
        gid = getattr(
            r,
            "growth_plant_id",
            None,
        )

        if not gid:
            continue

        rows.append(
            {
                "year": int(year),
                "plant_id": gid,
                "target": float(r.target),
                "target_source_file": str(p),
                "target_source_sheet": str(sheet),
                "target_source_id": str(r.plant_id),
                "target_rows_aggregated": int(r.target_rows),
                "id_mapping_method": str(r.mapping_method),
            }
        )

    chosen = {
        "year": year,
        "file": str(p),
        "sheet": str(sheet),
        "matched_target_ids": best["matched_target_ids"],
        "unique_target_ids": best["unique_target_ids"],
        "target_id_match_rate": best["target_id_match_rate"],
        "growth_target_coverage": best["growth_target_coverage"],
        "mapping_methods": best["mapping_methods"],
        "yield_col": best["yield_col"],
        "id_col": best["id_col"],
    }

    return (
        pd.DataFrame(rows),
        pd.DataFrame(audits),
        chosen,
    )


def build_three_year_manifest(cfg):
    root = Path(
        cfg["data"]["root"]
    )

    years = [
        int(x)
        for x in cfg["data"].get(
            "years",
            [2023, 2024, 2025],
        )
    ]

    out_dir = Path(
        cfg["data"].get(
            "manifest_output_dir",
            "./manifest_outputs",
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []
    growth_audits = []
    target_audits = []
    selected_targets = []
    mapping_audits = []
    year_summary = []

    overrides = (
        cfg.get("manifest", {})
        .get("target_overrides", {})
    )

    for year in years:
        print(
            f"\n[MANIFEST] Processing "
            f"{year} ..."
        )

        growth, ga = extract_growth_rows(
            root,
            year,
        )

        if len(ga):
            growth_audits.append(
                ga
            )

        if growth.empty:
            print(
                "[MANIFEST WARNING] "
                "No growth records "
                f"extracted for {year}."
            )

            year_summary.append(
                {
                    "year": year,
                    "growth_rows": 0,
                    "growth_plants": 0,
                    "target_ids": 0,
                    "matched_target_ids": 0,
                    "target_id_match_rate": 0.0,
                    "growth_target_coverage": 0.0,
                    "matched_rows": 0,
                    "matched_plants": 0,
                    "dates": 0,
                    "photo_coverage": 0.0,
                    "sensor_path_coverage": 0.0,
                }
            )
            continue

        growth_ids = sorted(
            growth["plant_id"]
            .dropna()
            .unique()
            .tolist()
        )

        override = (
            overrides.get(
                year,
                overrides.get(
                    str(year),
                    {},
                ),
            )
            or {}
        )

        (
            targets,
            ta,
            chosen,
        ) = extract_targets(
            root,
            year,
            growth_ids,
            override,
        )

        if len(ta):
            target_audits.append(
                ta
            )

        if chosen:
            selected_targets.append(
                chosen
            )

            print(
                f"[MANIFEST] {year} "
                "target source: "
                f"{Path(chosen['file']).name} / "
                f"{chosen['sheet']} / "
                f"{chosen['id_col']} -> "
                f"{chosen['yield_col']} / "
                "target-ID match="
                f"{chosen['target_id_match_rate']:.3f} / "
                "growth coverage="
                f"{chosen['growth_target_coverage']:.3f} / "
                "methods="
                f"{chosen['mapping_methods']}"
            )
        else:
            print(
                "[MANIFEST WARNING] "
                "No usable target table "
                f"selected for {year}."
            )

        target_cols = [
            "year",
            "plant_id",
            "target",
            "target_source_file",
            "target_source_sheet",
            "target_source_id",
            "target_rows_aggregated",
            "id_mapping_method",
        ]

        if not targets.empty:
            mapping_audits.append(
                targets[
                    [
                        "year",
                        "plant_id",
                        "target_source_id",
                        "id_mapping_method",
                    ]
                ].rename(
                    columns={
                        "plant_id":
                            "growth_plant_id",
                    }
                )
            )

        merged = growth.merge(
            targets[target_cols]
            if not targets.empty
            else pd.DataFrame(
                columns=target_cols
            ),
            on=[
                "year",
                "plant_id",
            ],
            how="left",
        )

        merged[
            "group"
        ] = (
            merged["year"]
            .astype(str)
            + "_"
            + merged["plant_id"]
            .astype(str)
        )

        matched = merged[
            merged["target"]
            .notna()
        ].copy()

        target_ids = (
            int(
                chosen[
                    "unique_target_ids"
                ]
            )
            if chosen
            else 0
        )

        matched_target_ids = (
            int(
                chosen[
                    "matched_target_ids"
                ]
            )
            if chosen
            else 0
        )

        target_id_match_rate = (
            float(
                chosen[
                    "target_id_match_rate"
                ]
            )
            if chosen
            else 0.0
        )

        growth_target_coverage = (
            float(
                chosen[
                    "growth_target_coverage"
                ]
            )
            if chosen
            else 0.0
        )

        summary = {
            "year":
                year,
            "growth_rows":
                int(
                    len(growth)
                ),
            "growth_plants":
                int(
                    growth[
                        "plant_id"
                    ].nunique()
                ),
            "target_ids":
                target_ids,
            "matched_target_ids":
                matched_target_ids,
            "target_id_match_rate":
                target_id_match_rate,
            "growth_target_coverage":
                growth_target_coverage,
            "matched_rows":
                int(
                    len(matched)
                ),
            "matched_plants":
                int(
                    matched[
                        "plant_id"
                    ].nunique()
                )
                if len(matched)
                else 0,
            "dates":
                int(
                    matched[
                        "date"
                    ].nunique()
                )
                if len(matched)
                else 0,
            "photo_coverage":
                float(
                    matched[
                        "photo_raw"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.len()
                    .gt(0)
                    .mean()
                )
                if len(matched)
                else 0.0,
            "sensor_path_coverage":
                float(
                    matched[
                        "sensor_raw"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.len()
                    .gt(0)
                    .mean()
                )
                if len(matched)
                else 0.0,
        }

        year_summary.append(
            summary
        )

        print(
            f"[MANIFEST] {year}: "
            f"growth_rows="
            f"{summary['growth_rows']}, "
            f"growth_plants="
            f"{summary['growth_plants']}, "
            f"target_ids="
            f"{summary['target_ids']}, "
            f"matched_target_ids="
            f"{summary['matched_target_ids']}, "
            f"target_ID_match="
            f"{summary['target_id_match_rate']:.3f}, "
            f"growth_target_coverage="
            f"{summary['growth_target_coverage']:.3f}, "
            f"matched_plants="
            f"{summary['matched_plants']}, "
            f"dates="
            f"{summary['dates']}, "
            f"photo_coverage="
            f"{summary['photo_coverage']:.3f}, "
            f"sensor_path_coverage="
            f"{summary['sensor_path_coverage']:.3f}"
        )

        if len(matched):
            all_rows.append(
                matched
            )

    if growth_audits:
        pd.concat(
            growth_audits,
            ignore_index=True,
            sort=False,
        ).to_csv(
            out_dir
            / "Growth_Source_Audit.csv",
            index=False,
        )

    if target_audits:
        pd.concat(
            target_audits,
            ignore_index=True,
            sort=False,
        ).to_csv(
            out_dir
            / "Target_Source_Audit.csv",
            index=False,
        )

    if mapping_audits:
        pd.concat(
            mapping_audits,
            ignore_index=True,
            sort=False,
        ).to_csv(
            out_dir
            / "Plant_ID_Harmonization_Audit.csv",
            index=False,
        )

    pd.DataFrame(
        selected_targets
    ).to_csv(
        out_dir
        / "Selected_Target_Sources.csv",
        index=False,
    )

    summary_df = pd.DataFrame(
        year_summary
    )

    summary_df.to_csv(
        out_dir
        / "Three_Year_Manifest_Audit.csv",
        index=False,
    )

    if not all_rows:
        raise RuntimeError(
            "No target-aligned "
            "multimodal rows were "
            "created. Inspect the "
            "manifest audit CSV files."
        )

    manifest = pd.concat(
        all_rows,
        ignore_index=True,
        sort=False,
    )

    base_cols = [
        "year",
        "date",
        "plant_id",
        "plant_id_raw",
        "group",
        "manual_csv",
        "growth_source",
        "photo_raw",
        "sensor_raw",
        "target",
        "target_source_file",
        "target_source_sheet",
        "target_source_id",
        "target_rows_aggregated",
        "id_mapping_method",
    ]

    tab_cols = sorted(
        [
            c
            for c in manifest.columns
            if c.startswith(
                "tab__"
            )
        ]
    )

    remaining = [
        c
        for c in manifest.columns
        if c
        not in base_cols
        + tab_cols
    ]

    manifest = manifest[
        [
            c
            for c in base_cols
            if c in manifest.columns
        ]
        + tab_cols
        + remaining
    ]

    manifest_path = Path(
        cfg[
            "data"
        ][
            "manifest_csv"
        ]
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    strict = bool(
        cfg[
            "data"
        ].get(
            "strict_three_year",
            True,
        )
    )

    present = sorted(
        manifest[
            "year"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    expected = sorted(
        years
    )

    # This validates whether the target source itself is harmonized to
    # growth IDs. It does NOT require every growth plant to have a yield
    # measurement, because the uploaded audit shows:
    # 2023: 39 target IDs among 90 growth IDs
    # 2025: 60 target IDs among 108 growth IDs.
    min_target_id_match = float(
        cfg.get(
            "manifest",
            {},
        ).get(
            "minimum_target_id_match_rate",
            cfg.get(
                "manifest",
                {},
            ).get(
                "minimum_target_match_rate",
                0.95,
            ),
        )
    )

    weak_years = (
        summary_df.loc[
            summary_df[
                "target_id_match_rate"
            ]
            .fillna(0)
            < min_target_id_match,
            "year",
        ]
        .astype(int)
        .tolist()
    )

    zero_target_years = (
        summary_df.loc[
            summary_df[
                "matched_target_ids"
            ]
            .fillna(0)
            <= 0,
            "year",
        ]
        .astype(int)
        .tolist()
    )

    min_target_ids = int(
        cfg.get(
            "manifest",
            {},
        ).get(
            "minimum_target_ids_per_year",
            1,
        )
    )

    small_target_years = (
        summary_df.loc[
            summary_df[
                "target_ids"
            ]
            .fillna(0)
            < min_target_ids,
            "year",
        ]
        .astype(int)
        .tolist()
    )

    print(
        "\n[MANIFEST] "
        "Final years:",
        present,
    )

    print(
        "[MANIFEST] "
        "Final rows:",
        len(
            manifest
        ),
    )

    print(
        "[MANIFEST] "
        "Final plant/year groups:",
        manifest[
            "group"
        ].nunique(),
    )

    print(
        "[MANIFEST] Saved:",
        manifest_path,
    )

    if strict:
        missing_years = sorted(
            set(expected)
            - set(present)
        )

        if (
            missing_years
            or weak_years
            or zero_target_years
            or small_target_years
        ):
            raise RuntimeError(
                "Strict three-year "
                "validation failed. "
                f"Missing years="
                f"{missing_years}; "
                "years below target-ID "
                f"match threshold="
                f"{weak_years}; "
                "years with zero matched "
                f"targets="
                f"{zero_target_years}; "
                "years below minimum target-ID count="
                f"{small_target_years}. "
                "Inspect "
                "Three_Year_Manifest_Audit.csv, "
                "Target_Source_Audit.csv, and "
                "Plant_ID_Harmonization_Audit.csv."
            )

    return (
        manifest,
        summary_df,
    )

