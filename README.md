Housing Price Prediction Model

This project builds and evaluates machine learning models to predict housing prices using the King County Housing dataset.
I structured it as a reproducible ML experiment pipeline, moving from exploratory analysis to repeatable training, evaluation, and experiment tracking, trying to mimic real-world machine learning workflows rather than one-off notebooks.

Project Goals
Predict home sale prices accurately
Using methods of handling price skew
Evaluate errors in log space and dollar space
Keep track of experiments so improvements are measurable
Build a project structure suitable for production-style ML workflows

Dataset
Source: King County Housing Data from Kaggle
Each row represents a home sale with features such as:
Bedrooms, bathrooms
Square footage (living area, basement)
Location (zipcode)
Sale date
Waterfront, view, condition, grade, etc.

Raw data lives in:
data/kc_house_data.csv

Project Structure
```
housing-price-model/
│
├── data/
│   └── kc_house_data.csv
│
├── notebooks/
│   └── exploration.ipynb        # EDA, feature testing, modeling experiments
│
├── src/
│   ├── features.py              # Data loading, cleaning, feature engineering
│   ├── train.py                 # Model training + persistence
│   ├── evaluate.py              # Metrics and error analysis
│   └── plots.py                 # (optional) visualization helpers
│
├── models/
│   └── *.pkl                    # Saved trained models
│
├── results/
│   ├── experiments.csv          # Master experiment log
│   └── <model_name__timestamp>/ # Per-run results
│       ├── metrics.json
│       └── report.md
│
├── run.py                       # End-to-end experiment runner
├── requirements.txt
└── README.md
```

Modeling Approach
1. Data Cleaning
Removed invalid rows (e.g., zero bedrooms or bathrooms)
Preserved valid zero values (e.g., basement size, waterfront, view indicators)

2. Feature Engineering
Extracted sale_year and sale_month from the sale date
Dropped identifiers (id, raw date)
Used log(price) as the modeling target to handle skewed price distributions

3. Models Tested
Linear Regression (baseline)
Linear Regression with one-hot encoded zipcodes
Random Forest Regressor (final model)

Model Validation & Evaluation
Data is split once into training and held-out test sets
Hyperparameters are selected using k-fold cross-validation on the training data only
Final metrics are reported once on the unseen test set
This provides a realistic estimate of generalization performance.

Metrics Reported
R² on log(price)
MAE (Mean Absolute Error) in dollars
RMSE in dollars
Median percent error
Median percent error by price tier
Evaluating both absolute and relative errors ensures performance across cheap, mid-range, and expensive homes.

Running an Experiment
From the project root:
python run.py
You’ll be prompted for a model name:
Model name (e.g. rf_v1_onehot_zip):

Each run automatically:
Trains the model
Saves the trained model to models/
Logs metrics to results/experiments.csv
Creates a per-run folder containing:
metrics.json (machine-readable metrics)
report.md (human-readable summary)

Example Results
Typical performance for the tuned Random Forest model:
R² (log price): ~0.89
Median percent error: ~8–9%
MAE: ~$65k–70k
Errors increase for the highest-priced homes, consistent with data scarcity and property uniqueness

Why This Structure?
This project is intentionally split into:
Exploration (notebooks) for rapid iteration and insight
Production-style code for reproducibility and clarity
Experiment tracking for measurable progress

This mirrors real-world ML workflows and makes it easy to:
Compare models
Tune hyperparameters responsibly
Add new features or algorithms
Resume work without invalidating prior results

Possible Extensions
Gradient boosting models (XGBoost / LightGBM)
Feature importance and partial dependence analysis
Price-tier–specific models
Deployment as an API or web application

Requirements
Install dependencies with:
pip install -r requirements.txt

Author
Built by Sean O'Donnell as a learning and portfolio project focused on:
Machine learning fundamentals
Error analysis and validation
Clean project structure
Reproducible experimentation