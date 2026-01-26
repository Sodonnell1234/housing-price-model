# Run everything from start to finish and log how well the model did
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


def slugify(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]+", "", name)
    return name or "unnamed_model"


def safe_json(obj):
    """Convert non-JSON types (Interval, np types, etc.) into JSON-friendly types."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[str(k)] = safe_json(v)  # force keys to str
        return out
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    # numpy scalars
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
    return f"{x*100:.1f}%"


def write_human_report(report_path: Path, model_name: str, metrics: dict) -> None:
    # expected keys (based on your current evaluate output)
    r2 = metrics.get("R2_log")
    mae = metrics.get("MAE_$")
    rmse = metrics.get("RMSE_$")
    med_pct = metrics.get("Median_pct_error")
    quint = metrics.get("Pct_error_quintiles_median", {})

    lines = []
    lines.append(f"# Experiment Report: {model_name}")
    lines.append("")
    lines.append("## Headline")
    if med_pct is not None:
        lines.append(f"- Median percent error: **{format_pct(med_pct)}** (half of predictions are within this % error).")
    if mae is not None:
        lines.append(f"- Typical dollar error (MAE): **{format_money(mae)}**.")
    if rmse is not None:
        lines.append(f"- Large-error sensitivity (RMSE): **{format_money(rmse)}**.")
    if r2 is not None:
        lines.append(f"- R² on log(price): **{r2:.4f}**.")
    lines.append("")

    lines.append("## Interpretation")
    if mae is not None:
        lines.append(f"- On average, the model is off by about **{format_money(mae)}**.")
    if med_pct is not None:
        lines.append(f"- Typical relative error is about **{format_pct(med_pct)}**, which is more comparable across cheap vs expensive homes.")
    if rmse is not None and mae is not None:
        lines.append(f"- RMSE being larger than MAE suggests there are some bigger misses/outliers (expected in housing).")
    lines.append("")

    if isinstance(quint, dict) and len(quint) > 0:
        lines.append("## Error by price tier (median % error)")
        # quint keys are Interval strings after safe_json, but we can print them
        for k, v in quint.items():
            try:
                lines.append(f"- {k}: **{format_pct(float(v))}**")
            except Exception:
                lines.append(f"- {k}: **{v}**")
        lines.append("")
        lines.append("### What this means")
        lines.append("- Middle tiers usually score best (more data + more consistent patterns).")
        lines.append("- The highest tier often has higher error (rarer homes, more unique features).")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def append_master_csv(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])

    if MASTER_CSV.exists():
        # handle empty file edge-case
        try:
            df_old = pd.read_csv(MASTER_CSV)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except pd.errors.EmptyDataError:
            df_all = df_new
    else:
        df_all = df_new

    df_all.to_csv(MASTER_CSV, index=False)


def main():
    # --- prompt for model name (PyCharm Run button friendly)
    raw_name = input("Model name (e.g. rf_v1_onehot_zip): ").strip()
    model_name = slugify(raw_name)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{model_name}__{run_id}"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{tag}.pkl"
    json_path = RESULTS_DIR / f"{tag}.json"
    report_path = RESULTS_DIR / f"{tag}.md"

    # --- Load + prep
    df = load_data("data/kc_house_data.csv")
    df = clean_data(df)
    df = engineer_features(df)
    X, y = split_features_target(df)

    # --- Train
    model, X_test, y_test = train_model(X, y)
    save_model(model, path=str(model_path))

    # --- Evaluate
    metrics = evaluate(model, X_test, y_test)
    metrics_json = safe_json(metrics)

    # --- Save per-run JSON + report
    json_path.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    write_human_report(report_path, model_name=tag, metrics=metrics_json)

    # --- Append master CSV (good for quick sorting)
    row = {
        "run_id": run_id,
        "model_name": model_name,
        "tag": tag,
        "model_file": str(model_path),
        # flat metrics for CSV
        "R2_log": metrics_json.get("R2_log"),
        "MAE_$": metrics_json.get("MAE_$"),
        "RMSE_$": metrics_json.get("RMSE_$"),
        "Median_pct_error": metrics_json.get("Median_pct_error"),
        # keep the detailed dict as JSON string
        "details_json": json.dumps(metrics_json.get("Pct_error_quintiles_median", {})),
        "report_file": str(report_path),
        "metrics_json_file": str(json_path),
    }
    append_master_csv(row)

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved report: {report_path}")
    print(f"Updated master log: {MASTER_CSV}")


if __name__ == "__main__":
    main()