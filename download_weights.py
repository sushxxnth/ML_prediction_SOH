"""
Downloads pre-trained model weights and verification result files from
the GitHub Release (v1.0.0) and installs them into the correct directories.

Requires the GitHub CLI (gh) to be authenticated:
    gh auth login

Run this once after cloning the repository:
    python3 download_weights.py

Then run the full verification:
    python3 REPRODUCE_PAPER_CLAIMS.py
"""

import subprocess
import zipfile
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO     = "sushxxnth/ML_prediction_SOH"
TAG      = "v1.0.0"

EXTRACT_MAP = {
    "pinn_causal_retrained.pt":              BASE_DIR / "reports/pinn_causal/pinn_causal_retrained.pt",
    "patt_best.pt":                          BASE_DIR / "reports/patt_classifier/patt_best.pt",
    "hero_model.pt":                         BASE_DIR / "reports/hero_model/hero_model.pt",
    "pinn_retrained_results.json":           BASE_DIR / "reports/pinn_causal/pinn_retrained_results.json",
    "zeroshot_baseline_comparison.json":     BASE_DIR / "reports/zeroshot_baseline_comparison.json",
    "patt_results.json":                     BASE_DIR / "reports/patt_classifier/patt_results.json",
    "counterfactual_validation_results.json": BASE_DIR / "reports/counterfactual_validation_results.json",
}


def check_gh():
    if shutil.which("gh") is None:
        print("ERROR: GitHub CLI (gh) not found.")
        print("Install: https://cli.github.com/  then run: gh auth login")
        sys.exit(1)
    result = subprocess.run(["gh", "auth", "status"], capture_output=True)
    if result.returncode != 0:
        print("ERROR: Not logged into gh CLI. Run: gh auth login")
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
    check_gh()

    tmp = BASE_DIR / ".release_tmp"
    tmp.mkdir(exist_ok=True)

    print(f"Downloading release assets ({TAG}) from {REPO}...")

    result = subprocess.run(
        ["gh", "release", "download", TAG,
         "--repo", REPO,
         "--dir", str(tmp),
         "--pattern", "*.zip"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"ERROR: {result.stderr.strip()}")
        sys.exit(1)

    downloaded = list(tmp.glob("*.zip"))
    if not downloaded:
        print("ERROR: No zip files downloaded.")
        sys.exit(1)

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
