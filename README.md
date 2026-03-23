# db-delay-forecaster

**Hadi Dawoud** · **GitHub:** https://github.com/HadiDawoud · **LinkedIn:** https://www.linkedin.com/in/hadidawoud

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-regression-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-boosting-01748F?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![Time series](https://img.shields.io/badge/evaluation%3A%20time--ordered%20holdout-555)](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
[![License: ISC](https://img.shields.io/badge/License-ISC-blue.svg)](LICENSE)

End-to-end **regression** pipeline: predict train **departure delay (minutes)** on open [Deutsche Bahn–style trip data](https://huggingface.co/datasets/piebro/deutsche-bahn-data). I built this to practice **honest time-series evaluation** (no random shuffling), strong **baselines**, and a clear story recruiters can skim in under a minute.

**Key metrics (holdout, best ML vs. mean baseline):** **MAE ≈ 3.55 min** · **R² ≈ 0.32** · **~22% lower MAE**

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.10+ |
| ML | scikit-learn (LinearRegression, RandomForest, HistGradientBoosting, `DummyRegressor`), LightGBM |
| Time-series validation | `TimeSeriesSplit` CV; temporal train → test split |
| Data | pandas, NumPy, Parquet (pyarrow), Hugging Face Hub |
| Viz | matplotlib, seaborn |
| API / ops (optional) | FastAPI, Docker |
| Dev | Ruff, pre-commit |

## Results (holdout, time-ordered test split)

Measured with `python main.py` and `MAX_ROWS=400000` (laptop-friendly; omit for full data). Data: Parquet **2024-09 … 2024-11** from [piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data).

| Model | MAE (min) | RMSE (min) | R² |
| --- | ---: | ---: | ---: |
| Baseline (mean) | 4.531 | 9.960 | &minus;0.002 |
| Baseline (lag-1) | 5.416 | 13.456 | &minus;0.828 |
| LinearRegression | 3.547 | 7.891 | 0.371 |
| **LightGBM** | **3.545** | **8.181** | **0.324** |
| RandomForest | 3.600 | 8.152 | 0.329 |
| HistGradientBoosting | 3.612 | 7.971 | 0.358 |

Best ML model (**LightGBM**): **21.8% lower MAE** vs. mean-only baseline on this run.

**CV on train only** (TimeSeriesSplit, `k=5`; MAE in minutes): LinearRegression **3.328** ±0.360, LightGBM **3.353** ±0.370, RandomForest **3.357** ±0.378, HistGradientBoosting **3.395** ±0.372, Baseline (mean) **4.261** ±0.385.

### Plots

| Model comparison (CV) | Feature importance |
| :---: | :---: |
| ![Model comparison](https://github.com/HadiDawoud/db-delay-forecaster/raw/main/outputs/plots/model_comparison.png) | ![Feature importance](https://github.com/HadiDawoud/db-delay-forecaster/raw/main/outputs/plots/feature_importance.png) |

| Delay distribution | Delay by hour |
| :---: | :---: |
| ![Delay distribution](https://github.com/HadiDawoud/db-delay-forecaster/raw/main/outputs/plots/delay_distribution.png) | ![Delay by hour](https://github.com/HadiDawoud/db-delay-forecaster/raw/main/outputs/plots/delay_by_hour.png) |

| Delay by weekday | Top stations |
| :---: | :---: |
| ![Delay by weekday](https://github.com/HadiDawoud/db-delay-forecaster/raw/main/outputs/plots/delay_by_weekday.png) | ![Top stations](https://github.com/HadiDawoud/db-delay-forecaster/raw/main/outputs/plots/top_stations.png) |

After training, `outputs/plots/holdout_predictions.png` is written as well.

## Problem & goal

**Target:** `delay_in_min` at **planned departure** — row-level regression on cleaned trips (not a full network simulator). Features include calendar signals, lags, and rolling stats; evaluation uses a **chronological** test set so metrics reflect drift, not shuffle leakage.

## Why it matters (short)

Realistic expected delay helps **connections**, **dispatch**, **passenger info**, and **KPIs**. This repo is a **portfolio / learning** pipeline on open data; production would add live feeds, route granularity, and governance.

## Deployment (API & Docker)

Optional serving after you train (`main.py`):

**FastAPI** — start the API, then open interactive docs:

```bash
python src/api.py
```

→ `http://localhost:8000/docs`

**Docker**

```bash
docker build -t db-delay-forecaster .
docker run -p 8000:8000 db-delay-forecaster
```

## How to run

Data: [piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data) (CC BY 4.0). Code: **ISC** ([`LICENSE`](LICENSE)).

```bash
git clone https://github.com/HadiDawoud/db-delay-forecaster.git
cd db-delay-forecaster
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # or: pip install -e .
python data/download.py
python eda.py
python main.py
```

Reproduce the table: `MAX_ROWS=400000 python main.py`. Outputs: console metrics (incl. baselines) and `outputs/plots/`.

**Dev (optional):** `pip install -e ".[dev]"`, `pre-commit install` — Ruff on commit.

Run lint/format on all files manually:

```bash
pre-commit run --all-files
```

### Baselines (context)

- **Mean:** `DummyRegressor` — training-set mean delay. ML should beat this or features are weak.
- **Lag-1:** previous row’s delay (persistence). Strong on autocorrelated series; ML should match or beat when extra signal helps.

### One prediction (`model.predict`)

Runnable: `python examples/predict_one.py` (from repo root, Parquet present). Minimal pattern:

```python
from pathlib import Path
import glob
import sys

ROOT = Path("/path/to/db-delay-forecaster")
sys.path.insert(0, str(ROOT / "src"))

from features import (
    FEATURE_COLS,
    TARGET_COL,
    TIME_COL,
    build_features,
    temporal_train_test_split,
)
from train import get_models

paths = sorted(glob.glob(str(ROOT / "data/raw/monthly_processed_data/*.parquet")))
df = build_features(paths)
train_df, test_df = temporal_train_test_split(df, time_col=TIME_COL, test_size=0.15)

model = get_models()["RandomForest"]
model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

sample = test_df.iloc[[0]]
pred = float(model.predict(sample[FEATURE_COLS])[0])
print(round(pred, 2), "min predicted delay")
```

## Author

**Hadi Dawoud** — same contact links as at the top of this README.

---

## Deutsch (Kurz)

**Ziel:** Verspätung in Minuten bei geplanter Abfahrt vorhersagen (Regression auf bereinigten Fahrten, Daten [piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data)). **Zeitlich aufgeteilter** Train/Test‑Split und Baselines (Mittelwert, Lag‑1). Hauptbefund wie oben: bestes ML (**LightGBM**) mit deutlich niedrigerem MAE als Mean‑Baseline. Ausführung: `pip install -r requirements.txt`, `python data/download.py`, `python main.py` (optional `MAX_ROWS=400000`).
