from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

(RAW_DIR / "monthly_processed_data").mkdir(parents=True, exist_ok=True)

months = ["2024-09", "2024-10", "2024-11"]

for month in months:
    print(f"Lade {month}...")
    hf_hub_download(
        repo_id="piebro/deutsche-bahn-data",
        repo_type="dataset",
        filename=f"monthly_processed_data/data-{month}.parquet",
        local_dir=str(RAW_DIR),
    )
    print(f"OK {month}")

print("\nAlle Dateien heruntergeladen.")
print(f"Pfad: {RAW_DIR / 'monthly_processed_data'}")
