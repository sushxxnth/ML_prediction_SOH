"""
Downloads pre-trained model weights and verification result files from
the GitHub Release (v1.0.0) and installs them into the correct directories.

Run this once after cloning the repository:
    python3 download_weights.py

Then run the full verification:
    python3 REPRODUCE_PAPER_CLAIMS.py
"""

import urllib.request
import zipfile
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Direct download URLs for the public GitHub release
ASSETS = [
    "https://github.com/sushxxnth/ML_prediction_SOH/releases/download/v1.0.0/model_weights.zip",
    "https://github.com/sushxxnth/ML_prediction_SOH/releases/download/v1.0.0/verification_results.zip"
]

EXTRACT_MAP = {
    "pinn_causal_retrained.pt":              BASE_DIR / "reports/pinn_causal/pinn_causal_retrained.pt",
    "patt_best.pt":                          BASE_DIR / "reports/patt_classifier/patt_best.pt",
    "hero_model.pt":                         BASE_DIR / "reports/hero_model/hero_model.pt",
    "pinn_retrained_results.json":           BASE_DIR / "reports/pinn_causal/pinn_retrained_results.json",
    "zeroshot_baseline_comparison.json":     BASE_DIR / "reports/zeroshot_baseline_comparison.json",
    "patt_results.json":                     BASE_DIR / "reports/patt_classifier/patt_results.json",
    "counterfactual_validation_results.json": BASE_DIR / "reports/counterfactual_validation_results.json",
}

def download_file(url: str, dest: Path):
    print(f"Downloading {url.split('/')[-1]}...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"ERROR: Failed to download {url}")
        print(f"Exception: {e}")
        sys.exit(1)

def extract(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            target = EXTRACT_MAP.get(name)
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"  Installed: {target.relative_to(BASE_DIR)}")

def main():
    tmp = BASE_DIR / ".release_tmp"
    tmp.mkdir(exist_ok=True)

    print("Fetching release assets (v1.0.0)...")

    downloaded = []
    for i, url in enumerate(ASSETS):
        zip_path = tmp / f"asset_{i}.zip"
        download_file(url, zip_path)
        downloaded.append(zip_path)

    for zip_path in downloaded:
        print(f"  Extracting {zip_path.name} ...")
        extract(zip_path)

    # Cleanup
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()

    print("\nAll weights and results installed.")
    print("Run:  python3 REPRODUCE_PAPER_CLAIMS.py")


if __name__ == "__main__":
    main()
