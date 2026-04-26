import pandas as pd

FEATURE_COLS = [
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "is_rush_hour",
    "lag_1",
    "lag_3",
    "lag_7",
    "rolling_mean_5",
    "rolling_std_5",
]

TARGET_COL = "delay_in_min"
TIME_COL = "departure_planned_time"


def temporal_train_test_split(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    split_at = int(n * (1 - test_size))
    if split_at >= n:
        raise ValueError(f"test_size={test_size} too large - no test data")
    train_df = df.iloc[:split_at].copy()
    test_df = df.iloc[split_at:].copy()
    return train_df, test_df


def load_data(paths: list) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError(
            "No parquet files in data/raw/monthly_processed_data/. Run python data/download.py."
        )
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Geladen: {df.shape[0]:,} Zeilen")
    return df


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["is_canceled"].eq(False)].copy()
    df = df[df["delay_in_min"].notna()].copy()
    df = df[df["delay_in_min"].between(-10, 180)]
    df = df.sort_values("departure_planned_time").reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["departure_planned_time"])
    df["hour"] = t.dt.hour
    df["weekday"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_rush_hour"] = t.dt.hour.isin([7, 8, 17, 18]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df["lag_1"] = df[TARGET_COL].shift(1)
    df["lag_3"] = df[TARGET_COL].shift(3)
    df["lag_7"] = df[TARGET_COL].shift(7)
    df["rolling_mean_5"] = df[TARGET_COL].rolling(5).mean()
    df["rolling_std_5"] = df[TARGET_COL].rolling(5).std()
    return df


def build_features(paths: list, max_rows: int | None = None) -> pd.DataFrame:
    df = load_data(paths)
    df = filter_and_clean(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) == 0:
        raise ValueError("No valid data after feature engineering. Check data quality.")
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].copy().reset_index(drop=True)
    return df
