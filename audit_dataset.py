from pathlib import Path
import pandas as pd
from src.utils import load_config
from src.manifest import discover_growth_files, discover_target_files


def main():
    cfg = load_config("config.yaml")
    root = Path(cfg["data"]["root"])
    rows = []
    for year in cfg["data"].get("years", [2023, 2024, 2025]):
        growth = discover_growth_files(root, int(year))
        targets = discover_target_files(root, int(year))
        rows.append({
            "year": int(year), "growth_files": len(growth), "target_files": len(targets),
            "growth_file_examples": " ; ".join(str(p.relative_to(root)) for p in growth[:10]),
            "target_file_examples": " ; ".join(str(p.relative_to(root)) for p in targets[:10]),
        })
    out = pd.DataFrame(rows)
    outdir = Path(cfg["data"]["manifest_output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "Dataset_Discovery_Audit.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print("\nSaved:", path)


if __name__ == "__main__":
    main()
