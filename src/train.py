# Training models with the engineered features

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_cols = ["zipcode"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),])

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=25,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1)

    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)])

    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test

def save_model(model, path="models/model.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)