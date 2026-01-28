# run.py
# Run everything end-to-end, save model + per-run artifacts, and append a master experiments.csv

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import re

import pandas as pd

from src.features import load_data, clean_data, engineer_features, split_features_target
from src.train import train_model, save_model
from src.evaluate import evaluate


RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
MASTER_CSV = RESULTS_DIR / "experiments.csv"


# -------------------------
# Helpers
# -------------------------
def slugify(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]+", "", name)
    return name or "unnamed_model"


def safe_json(obj):
    # Convert non-JSON types (Interval, numpy scalars, etc.) to JSON-friendly types.
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}  # force keys to str
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return obj


def format_money(x: float) -> str:
    return f"${x:,.0f}"


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_report(report_path: Path, model_tag: str, metrics: dict) -> None:

    # Writes a readable markdown report for this run.
    # Expects evaluate() to return keys like:
    # R2_log, MAE_$, RMSE_$, Median_pct_error, Pct_error_quintiles_median

    r2 = metrics.get("R2_log")
    mae = metrics.get("MAE_$")
    rmse = metrics.get("RMSE_$")
    med_pct = metrics.get("Median_pct_error")
    quint = metrics.get("Pct_error_quintiles_median", {})

    lines: list[str] = []
    lines.append(f"# Experiment Report: {model_tag}")
    lines.append("")

    lines.append("## Summary")
    if med_pct is not None:
        lines.append(f"- Median percent error: **{format_pct(float(med_pct))}** (half the predictions are within this % error).")
    if mae is not None:
        lines.append(f"- Typical dollar error (MAE): **{format_money(float(mae))}**.")
    if rmse is not None:
        lines.append(f"- RMSE: **{format_money(float(rmse))}** (penalizes big misses more).")
    if r2 is not None:
        lines.append(f"- R² on log(price): **{float(r2):.4f}**.")
    lines.append("")

    lines.append("## Interpretation")
    if mae is not None:
        lines.append(f"- On average, the model misses by about **{format_money(float(mae))}**.")
    if med_pct is not None:
        lines.append(f"- Typical relative error is about **{format_pct(float(med_pct))}** (better for comparing cheap vs expensive homes).")
    if rmse is not None and mae is not None:
        lines.append("- RMSE being noticeably larger than MAE usually means a few larger outliers (common in housing).")
    lines.append("")

    if isinstance(quint, dict) and len(quint) > 0:
        lines.append("## Error by price tier (median % error)")
        for k, v in quint.items():
            try:
                lines.append(f"- {k}: **{format_pct(float(v))}**")
            except Exception:
                lines.append(f"- {k}: **{v}**")
        lines.append("")
        lines.append("### Notes")
        lines.append("- Middle tiers often do best (more data + more consistent patterns).")
        lines.append("- Highest tier often has higher error (rarer + more unique homes).")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def append_master_csv(row: dict) -> None:
    # Appends a single row to results/experiments.csv (creates it if missing).
    # Handles the 'empty file' case cleanly.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([row])

    if MASTER_CSV.exists():
        try:
            df_old = pd.read_csv(MASTER_CSV)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except pd.errors.EmptyDataError:
            df_all = df_new
    else:
        df_all = df_new

    df_all.to_csv(MASTER_CSV, index=False)


# -------------------------
# Main
# -------------------------
def main() -> None:
    # Prompt for model name (works great with PyCharm Run button)
    raw_name = input("Model name (e.g. rf_v1_onehot_zip): ").strip()
    model_name = slugify(raw_name)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{model_name}__{run_id}"

    # Create a per-run folder: results/<tag>/
    run_dir = RESULTS_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{tag}.pkl"
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "report.md"

    # Load + prep
    df = load_data("data/kc_house_data.csv")
    df = clean_data(df)
    df = engineer_features(df)
    X, y = split_features_target(df)

    # Train
    model, X_test, y_test = train_model(X, y)
    save_model(model, path=str(model_path))

    # Evaluate
    metrics = evaluate(model, X_test, y_test)
    metrics_json = safe_json(metrics)

    # Save per-run artifacts
    metrics_path.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    write_report(report_path, model_tag=tag, metrics=metrics_json)

    # Append master CSV (easy sorting/search)
    row = {
        "run_id": run_id,
        "model_name": model_name,
        "tag": tag,
        "model_file": str(model_path),
        "results_dir": str(run_dir),
        # Flat metrics for quick filtering in CSV
        "R2_log": metrics_json.get("R2_log"),
        "MAE_$": metrics_json.get("MAE_$"),
        "RMSE_$": metrics_json.get("RMSE_$"),
        "Median_pct_error": metrics_json.get("Median_pct_error"),
        # Detailed dict stored as JSON string:
        "details_json": json.dumps(metrics_json.get("Pct_error_quintiles_median", {})),
        "report_file": str(report_path),
        "metrics_json_file": str(metrics_path),
    }
    append_master_csv(row)

    print(f"\nRun complete: {tag}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved report: {report_path}")
    print(f"Updated master log: {MASTER_CSV}")


if __name__ == "__main__":
    main()