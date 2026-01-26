# Run everything from start to finish and log how well the model did

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from src.features import load_data, clean_data, engineer_features, split_features_target
from src.train import train_model
from src.evaluate import evaluate
from pandas.errors import EmptyDataError

RESULTS_PATH = Path("results/experiments.csv")
MODELS_DIR = Path("models")


def append_results(row: dict):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([row])

    if RESULTS_PATH.exists():
        try:
            df_old = pd.read_csv(RESULTS_PATH)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except EmptyDataError:
            # File exists but is empty
            df_all = df_new
    else:
        df_all = df_new

    df_all.to_csv(RESULTS_PATH, index=False)

def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, np.generic):   # numpy scalar
        return obj.item()
    return obj

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Load + prep
    df = load_data("data/kc_house_data.csv")
    df = clean_data(df)
    df = engineer_features(df)

    X, y = split_features_target(df)

    # --- Train
    model, X_test, y_test = train_model(X, y)

    # --- Evaluate
    metrics = evaluate(model, X_test, y_test)

    # --- Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"rf_pipeline_{run_id}.pkl"
    joblib.dump(model, model_path)

    # --- Log results (flatten anything complex)
    row = {
        "run_id": run_id,
        "model_file": str(model_path),
        **{k: v for k, v in metrics.items() if not isinstance(v, dict)},
        "details_json": json.dumps(json_safe({k: v for k, v in metrics.items() if isinstance(v, dict)}))
    }

    append_results(row)

    print("Saved model:", model_path)
    print("Logged results to:", RESULTS_PATH)
    print(metrics)


if __name__ == "__main__":
    main()