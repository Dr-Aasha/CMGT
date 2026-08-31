from src.utils import load_config
from src.manifest import build_three_year_manifest


def main():
    cfg = load_config("config.yaml")
    manifest, audit = build_three_year_manifest(cfg)
    print("\nThree-year manifest build completed.")
    print(audit.to_string(index=False))
    print("\nManifest rows:", len(manifest))
    print("Years:", sorted(manifest["year"].astype(int).unique().tolist()))
    print("Plant/year groups:", manifest["group"].nunique())


if __name__ == "__main__":
    main()
