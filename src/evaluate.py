import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred, model_name: str = "", verbose: bool = True) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if verbose:
        print(f"{'=' * 45}")
        print(f"  {model_name}")
        print(f"  MAE:  {mae:.2f} min")
        print(f"  RMSE: {rmse:.2f} min")
        print(f"  R²:   {r2:.3f}")
        print(f"{'=' * 45}")

    return {
        "model": model_name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3),
    }
