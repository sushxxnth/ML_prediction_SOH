"""
Downloads pre-trained model checkpoints and released result artifacts from the
GitHub Release and installs them under reports/ (which is gitignored).

Run this once after cloning the repository:
    python3 download_weights.py

Then verify every reported number at once:
    python3 reproduce_paper_results.py

The two release archives store their files with paths relative to the repository
root (e.g. reports/zeroshot_table_rebuild.json), so extraction simply mirrors
them into place -- no per-file mapping to maintain.
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RELEASE = "v1.1.0"
BASE_URL = f"https://github.com/sushxxnth/ML_prediction_SOH/releases/download/{RELEASE}"
ASSETS = ["model_weights.zip", "verification_results.zip"]


def download(url: str, dest: Path):
    print(f"  downloading {url.split('/')[-1]} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: failed to download {url}\n  {e}")
        sys.exit(1)


def extract(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            target = BASE_DIR / name
            # keep everything inside the repo (defensive against absolute/.. paths)
            if not str(target.resolve()).startswith(str(BASE_DIR.resolve())):
                print(f"  skipping unsafe path: {name}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"    installed {name}")


def main():
    tmp = BASE_DIR / ".release_tmp"
    tmp.mkdir(exist_ok=True)
    print(f"Fetching release assets ({RELEASE}) ...")
    for asset in ASSETS:
        zip_path = tmp / asset
        download(f"{BASE_URL}/{asset}", zip_path)
        print(f"  extracting {asset} ...")
        extract(zip_path)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    print("\nAll checkpoints and result artifacts installed under reports/.")
    print("Now run:  python3 reproduce_paper_results.py")


if __name__ == "__main__":
    main()
