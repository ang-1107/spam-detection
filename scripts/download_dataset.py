import argparse
import os
import zipfile
from pathlib import Path

def _require_kaggle_auth():
    # Kaggle package will read ~/.kaggle/kaggle.json or env vars.
    # Give a clear error if neither looks present.
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env = bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))
    if not kaggle_json.exists() and not has_env:
        raise RuntimeError(
            "Kaggle credentials not found.\n"
            "Set up one of:\n"
            "  1) ~/.kaggle/kaggle.json (recommended)\n"
            "  2) environment variables KAGGLE_USERNAME and KAGGLE_KEY\n"
        )

def main():
    p = argparse.ArgumentParser(description="Download SMS Spam Collection dataset from Kaggle.")
    p.add_argument("--dataset", default="uciml/sms-spam-collection-dataset")
    p.add_argument("--out-dir", default="data/raw", help="Directory to store downloaded dataset files")
    p.add_argument("--force", action="store_true", help="Re-download even if files already exist")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The Kaggle dataset typically provides a zip that contains spam.csv (windows-1252)
    spam_csv = out_dir / "spam.csv"
    zip_path = out_dir / "sms-spam-collection.zip"

    if spam_csv.exists() and not args.force:
        print(f"Dataset already present: {spam_csv}")
        return

    _require_kaggle_auth()

    from kaggle.api.kaggle_api_extended import KaggleApi  # imported after auth check

    api = KaggleApi()
    api.authenticate()

    if zip_path.exists() and args.force:
        zip_path.unlink(missing_ok=True)

    print(f"Downloading Kaggle dataset: {args.dataset}")
    api.dataset_download_files(args.dataset, path=str(out_dir), unzip=False)

    # Kaggle names the downloaded file after the dataset slug (not always stable),
    # so find the newest zip in out_dir.
    zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise RuntimeError(f"No zip file found in {out_dir} after download.")
    downloaded_zip = zips[0]

    # Normalize zip name to something stable (optional)
    if downloaded_zip != zip_path:
        downloaded_zip.replace(zip_path)

    print(f"Unzipping: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # Ensure spam.csv exists
    if not spam_csv.exists():
        # Some datasets extract into subfolders; try to locate spam.csv
        found = list(out_dir.rglob("spam.csv"))
        if found:
            found[0].replace(spam_csv)

    if not spam_csv.exists():
        raise RuntimeError(
            "Download completed but spam.csv was not found after unzip. "
            f"Check contents of {out_dir}."
        )

    print(f"Ready: {spam_csv}")

if __name__ == "__main__":
    main()