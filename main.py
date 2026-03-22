import glob
import os
import sys
import joblib
from pathlib import Path

sys.path.append("src")

# Optional: cap rows for faster local runs (same pipeline). Example: MAX_ROWS=400000
_MAX = os.environ.get("MAX_ROWS")
MAX_ROWS = int(_MAX) if _MAX and _MAX.isdigit() else None

from features import (
    FEATURE_COLS,
    TARGET_COL,
    TIME_COL,
    build_features,
    temporal_train_test_split,
)
from train import (
    ML_MODEL_NAMES,
    compare_models,
    evaluate_models_holdout,
    get_models,
    plot_feature_importance,
    plot_holdout_predictions,
    plot_model_comparison,
    print_improvement_vs_mean_baseline,
    print_top_feature_importances,
)

ROOT = Path(__file__).resolve().parent
HOLDOUT_FRACTION = 0.15

os.makedirs(ROOT / "outputs/plots", exist_ok=True)

print("=== DB Delay Forecaster ===\n")

paths = sorted(glob.glob(str(ROOT / "data/raw/monthly_processed_data/*.parquet")))
print(f"Lade {len(paths)} Dateien...")
if MAX_ROWS is not None:
    print(f"(MAX_ROWS={MAX_ROWS:,} — voller Lauf ohne diese Variable)\n")
df = build_features(paths, max_rows=MAX_ROWS)
print(f"Features gebaut: {df.shape[0]:,} Zeilen\n")

train_df, test_df = temporal_train_test_split(df, time_col=TIME_COL, test_size=HOLDOUT_FRACTION)
t_train = train_df[TIME_COL]
t_test = test_df[TIME_COL]
print(
    f"Train (ca. {100 * (1 - HOLDOUT_FRACTION):.0f}%): "
    f"{t_train.min()} → {t_train.max()}  ({len(train_df):,} Zeilen)"
)
print(
    f"Test  (ca. {100 * HOLDOUT_FRACTION:.0f}%):  "
    f"{t_test.min()} → {t_test.max()}  ({len(test_df):,} Zeilen)\n"
)

print("Modellvergleich – TimeSeriesSplit CV (nur Train)...\n")
cv_results = compare_models(train_df)
plot_model_comparison(cv_results, str(ROOT / "outputs/plots/model_comparison.png"))
plot_feature_importance(train_df, str(ROOT / "outputs/plots/feature_importance.png"))

print("\nHoldout – Train → Test (letzter Zeitblock):\n")
holdout_df = evaluate_models_holdout(train_df, test_df)
print(holdout_df.to_string(index=False))
print_improvement_vs_mean_baseline(holdout_df)
print_top_feature_importances(train_df, n=5)

holdout_ml = holdout_df[holdout_df["model"].isin(ML_MODEL_NAMES)]
best_name = holdout_ml.loc[holdout_ml["MAE"].idxmin(), "model"]
best_model = get_models()[best_name]
best_model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

model_path = ROOT / "outputs" / "model.joblib"
joblib.dump(best_model, model_path)
print(f"\nBestes Modell gespeichert: {model_path}")

y_hat = best_model.predict(test_df[FEATURE_COLS])
plot_holdout_predictions(
    test_df[TARGET_COL].to_numpy(),
    y_hat,
    str(ROOT / "outputs/plots/holdout_predictions.png"),
    title=f"Holdout – bestes Modell: {best_name}",
)

print("\n=== CV (Train) ===")
print(cv_results.to_string(index=False))
print(f"\nBestes Modell (Holdout-MAE): {best_name}")
print("\nFertig. Plots in outputs/plots/")
