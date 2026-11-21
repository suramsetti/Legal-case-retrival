import os
import shutil
import json
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "Exploration-Lab/IL-TUR"
REPO_TYPE = "dataset"

SPLIT_FILES = {
    "train_queries": "pcr/train_queries-00000-of-00001.parquet",
    "train_candidates": "pcr/train_candidates-00000-of-00001.parquet",
    "dev_queries": "pcr/dev_queries-00000-of-00001.parquet",
    "dev_candidates": "pcr/dev_candidates-00000-of-00001.parquet",
    "test_queries": "pcr/test_queries-00000-of-00001.parquet",
    "test_candidates": "pcr/test_candidates-00000-of-00001.parquet",
}

BASE_OUTPUT_DIR = Path("data/external/IL-TUR-PCR")

def download_and_organize():
    downloaded_paths = {}
    print("Beginning download of IL-TUR PCR splits...")
    for split_name, remote_path in SPLIT_FILES.items():
        print(f"Downloading {split_name} ({remote_path})...")
        local_cache_file = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            repo_type=REPO_TYPE
        )
        target_dir = BASE_OUTPUT_DIR / split_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{split_name}.parquet"
        shutil.copyfile(local_cache_file, target_file)
        downloaded_paths[split_name] = target_file
        print(f"Saved {split_name} to {target_file} ({os.path.getsize(target_file)/(1024*1024):.2f} MB)")
    return downloaded_paths

if __name__ == "__main__":
    download_and_organize()

