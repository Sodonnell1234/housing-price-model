Housing Price Prediction Model

This project builds and evaluates machine learning models to predict housing prices using the King County housing dataset.
It is structured as a reproducible ML experiment pipeline, moving from exploratory analysis to repeatable training, evaluation, and experiment tracking.

Project Goals
Predict home sale prices as accurately as possible
Handle price skew using log transformations
Evaluate errors both in log space and real dollar space
Track experiments cleanly so improvements are measurable
Build a project structure suitable for real ML workflows (not just notebooks)

Dataset
Source: King County Housing Data
Each row represents a home sale with features such as:
Bedrooms, bathrooms
Square footage (living, basement)
Location (zipcode)
Sale date
Waterfront, view, condition, grade, etc.

Raw data lives in:
data/kc_house_data.csv

Project Structure
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

Modeling Approach
1. Data Cleaning
Removed invalid rows (e.g. zero bedrooms or bathrooms)
Preserved valid zero values (e.g. basement size, waterfront)

2. Feature Engineering
Extracted sale_year and sale_month from the sale date
Dropped identifiers (id, raw date)
Used log(price) as the modeling target to handle skew

3. Models Tested
Linear Regression (baseline)
Linear Regression with one-hot encoded zipcodes
Random Forest Regressor (final model)

4. Evaluation Strategy

Models are evaluated using:
R² on log(price)
MAE (Mean Absolute Error) in dollars
RMSE in dollars
Median percent error
Median percent error by price tier
This ensures performance is understood across cheap, mid-range, and expensive homes.

Running an Experiment
From the project root:
python run.py
You’ll be prompted for a model name:
Model name (e.g. rf_v1_onehot_zip):

Each run automatically:
Trains the model
Saves the model to models/
Logs metrics to results/experiments.csv
Creates a per-run folder with:
metrics.json (machine-readable)
report.md (intuitive summary)
Example Results

Typical performance for the Random Forest model:
R² (log price): ~0.89
Median percent error: ~8–9%
MAE: ~$65k–70k
Errors increase for the highest-priced homes (expected due to rarity and uniqueness)

Why This Structure?

This project is intentionally split into:
Exploration (notebooks) → rapid iteration and insight
Production-style code (src/) → reproducibility and clarity
Experiment tracking (results/) → measurable progress

This mirrors real-world ML workflows and makes it easy to:
Compare models
Tune hyperparameters
Add new features or algorithms
Resume work without breaking past results

Possible Next Steps:
Add cross-validation
Hyperparameter search (GridSearch / RandomizedSearch)
Gradient Boosting models (XGBoost / LightGBM)
Feature importance visualization
Price-tier–specific models
Deployment as an API or web app

Requirements
Install dependencies with:
pip install -r requirements.txt

Author
Built by Sean O'Donnell as a learning and portfolio project focused on:
Machine learning fundamentals
Error analysis
Clean project structure
Reproducible experimentation