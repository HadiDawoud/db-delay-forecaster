"""
Minimal example: train on historical data, predict delay (minutes) for one holdout row.
Run from repo root: python examples/predict_one.py
Requires data under data/raw/monthly_processed_data/*.parquet (see data/download.py).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (  # noqa: E402
    FEATURE_COLS,
    TARGET_COL,
    TIME_COL,
    build_features,
    temporal_train_test_split,
)
from train import get_models  # noqa: E402

HOLDOUT_FRAC = 0.15


def main() -> None:
    paths = sorted(
        glob.glob(str(ROOT / "data/raw/monthly_processed_data/*.parquet"))
    )
    if not paths:
        raise SystemExit(
            "No parquet files. From repo root run: python data/download.py"
        )

    df = build_features(paths)
    train_df, test_df = temporal_train_test_split(
        df, time_col=TIME_COL, test_size=HOLDOUT_FRAC
    )

    model = get_models()["RandomForest"]
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

    sample = test_df.iloc[[0]]
    predicted_min = float(model.predict(sample[FEATURE_COLS])[0])
    actual_min = float(sample[TARGET_COL].iloc[0])

    print("Single-row prediction (first holdout departure)")
    print(f"  predicted delay: {predicted_min:.2f} min")
    print(f"  actual delay:      {actual_min:.2f} min")


if __name__ == "__main__":
    main()
