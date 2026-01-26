# Test the trained models on performance metrics and evaluate

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    pred_price = np.expm1(preds)
    true_price = np.expm1(y_test)

    mae = mean_absolute_error(true_price, pred_price)
    rmse = np.sqrt(mean_squared_error(true_price, pred_price))

    return {
        "MAE": mae,
        "RMSE": rmse
    }