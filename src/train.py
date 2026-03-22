import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import evaluate as eval_report
from features import FEATURE_COLS, TARGET_COL

RANDOM_STATE = 42


def get_models() -> dict:
    return {
        "LinearRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.05,
            n_iter_no_change=10,
        ),
    }


def compare_models(df_train: pd.DataFrame) -> pd.DataFrame:
    X = df_train[FEATURE_COLS]
    y = df_train[TARGET_COL]
    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for name, model in get_models().items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=tscv,
            scoring="neg_mean_absolute_error",
        )
        mae = -scores.mean()
        std = scores.std()
        print(f"{name:25s}  MAE = {mae:.2f} (+/- {std:.2f})")
        results.append(
            {
                "model": name,
                "MAE": round(mae, 3),
                "std": round(std, 3),
            }
        )

    return pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)


def evaluate_models_holdout(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]
    rows = []

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = eval_report(
            y_test, y_pred, model_name=name, verbose=False
        )
        rows.append(metrics)

    return pd.DataFrame(rows)


def plot_model_comparison(results: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        results["model"],
        results["MAE"],
        xerr=results["std"],
        color="steelblue",
        capsize=5,
    )
    ax.set_xlabel("MAE (Minuten)")
    ax.set_title("Modellvergleich – CV auf Train (TimeSeriesSplit)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"saved: {output_path}")


def plot_feature_importance(df: pd.DataFrame, output_path: str):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    model = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X, y)

    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance – Random Forest (Train)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"saved: {output_path}")


def plot_holdout_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    title: str,
    max_points: int = 15_000,
):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.15, s=8, color="steelblue")
    lim = max(float(y_true.max()), float(y_pred.max()), 1.0)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Ideal")
    ax.set_xlabel("Tatsächliche Verspätung (min)")
    ax.set_ylabel("Vorhersage (min)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"saved: {output_path}")
