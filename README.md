# Housing Price Prediction Model

I built this to practice structuring an ML project like an actual experiment rather than a throwaway notebook — tracked runs, held-out test sets, the whole thing. Uses King County housing data from Kaggle.

The goal wasn't just to get a good R² — it was to build something reproducible where I could actually compare model versions and know whether a change helped or hurt.

---

## Dataset

[King County Housing Data](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) — home sales in King County, WA with features like square footage, location (zipcode), condition, grade, waterfront, and sale date.

Raw data: `data/kc_house_data.csv`

---

## Project Structure

```
housing-price-model/
│
├── data/
│   └── kc_house_data.csv
│
├── notebooks/
│   └── exploration.ipynb        # EDA, feature testing, early modeling
│
├── src/
│   ├── features.py              # Data loading, cleaning, feature engineering
│   ├── train.py                 # Model training + saving
│   ├── evaluate.py              # Metrics and error analysis
│   └── plots.py                 # Visualization helpers
│
├── models/
│   └── *.pkl                    # Saved trained models
│
├── results/
│   ├── experiments.csv          # Experiment log across all runs
│   └── <model_name__timestamp>/
│       ├── metrics.json
│       └── report.md
│
├── run.py                       # End-to-end runner
├── requirements.txt
└── README.md
```

---

## Approach

Dropped rows with zero bedrooms or bathrooms (bad data). Kept zero values for basement, waterfront, and view

Extracted `sale_year` and `sale_month` from the date column, dropped raw date and ID. Modeled `log(price)` instead of raw price to deal with the right skew in home values.

- Linear Regression (baseline)
- Linear Regression + one-hot encoded zipcodes
- Random Forest (final)

Hyperparameters tuned with k-fold CV on training data only. Final metrics evaluated once on a held-out test set.

---

## Results

Best performance (tuned Random Forest):

| Metric | Value |
|---|---|
| R² (log price) | ~0.89 |
| Median % error | ~8–9% |
| MAE | ~$65k–70k |

Errors are higher at the top end of the price range which is expected since high-end homes are sparse and genuinely harder to predict.

---

## Running It

```bash
pip install -r requirements.txt
python run.py
```

You get prompted for a model name (e.g. `rf_v1_onehot_zip`). Each run saves the model, logs metrics to `results/experiments.csv`, and writes a per-run folder with a JSON metrics file and a markdown summary.

---

## Author

Sean O'Donnell