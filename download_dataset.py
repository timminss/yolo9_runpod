import os
import sys
import zipfile
from pathlib import Path

import requests


def download_and_extract(url: str, dest_root: Path) -> Path:
    dest_root.mkdir(parents=True, exist_ok=True)
    zip_path = dest_root / "dataset.zip"

    print(f"[download_dataset] Downloading dataset from {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"[download_dataset] Download complete, saved to {zip_path}")

    extract_dir = dest_root / "dataset"
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download_dataset] Extracting into {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Many Roboflow exports put everything in a single top-level directory.
    # We keep the extraction root stable and let the training script point
    # at data.yaml under this folder.
    print(f"[download_dataset] Extraction finished")
    return extract_dir


def main() -> None:
    url = os.getenv("ROBOFLOW_DATASET_URL")
    if not url:
        print("[download_dataset] ROBOFLOW_DATASET_URL is not set", file=sys.stderr)
        sys.exit(1)

    data_root = Path(os.getenv("DATA_ROOT", "/workspace/data"))
    extract_dir = download_and_extract(url, data_root)

    # Heuristic: try to report a likely data.yaml path for downstream scripts.
    candidate = next(extract_dir.rglob("data.yaml"), None)
    if candidate:
        print(f"[download_dataset] Found data.yaml at: {candidate}")
    else:
        print("[download_dataset] WARNING: No data.yaml found under extracted dataset", file=sys.stderr)


if __name__ == "__main__":
    main()

