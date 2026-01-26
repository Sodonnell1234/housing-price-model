# Test the trained models on performance metrics and evaluate
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    pred_price = np.expm1(preds)
    true_price = np.expm1(y_test)

    mae = mean_absolute_error(true_price, pred_price)
    rmse = np.sqrt(mean_squared_error(true_price, pred_price))
    r2 = r2_score(y_test, preds)  # in log space, consistent with training target

    abs_error = np.abs(pred_price - true_price)
    pct_error = abs_error / true_price

    # error by price quintile
    bins = pd.qcut(true_price, q=5, duplicates="drop")
    bucket_median_pct = (
        pd.DataFrame({"true_price": true_price, "pct_error": pct_error})
        .groupby(bins)["pct_error"]
        .median()
    )

    return {
        "R2_log": float(r2),
        "MAE_$": float(mae),
        "RMSE_$": float(rmse),
        "Median_pct_error": float(np.median(pct_error)),
        "Pct_error_quintiles_median": bucket_median_pct.to_dict(),}