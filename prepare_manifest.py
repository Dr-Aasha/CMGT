from pathlib import Path
import shutil

from src.utils import load_config
from src.manifest import build_three_year_manifest


def main():
    cfg = load_config("config.yaml")
    target = Path(cfg["data"]["manifest_csv"])
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print("Manifest already exists:", target)
        return

    for candidate in cfg["data"].get("external_manifest_candidates", []):
        p = Path(candidate)
        if p.exists():
            shutil.copy2(p, target)
            print("Reused verified three-year manifest:")
            print(" ", p)
            print(" ->", target)
            return

    print("No external verified manifest found; rebuilding from raw Horti-M3 files.")
    build_three_year_manifest(cfg)


if __name__ == "__main__":
    main()
