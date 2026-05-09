# Training the models with the engineered features
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV


def build_pipeline(X) -> Pipeline:

    # build preprocessing and the model pipleine
    # one-hot encode zipcode, pass through numeric columns

    cat_cols = [c for c in ["zipcode"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            ("model", model),
        ]
    )
    return pipe


def train_model(
    X_train,
    y_train,
    *,
    tune: bool = False,
    cv_folds: int = 5,
    n_iter: int = 30,
    random_state: int = 42,
) -> Tuple[Any, Dict]:

    # train model on X_train/y_train, if tune = True, uses RandomizedSearchCV with K-fold CV on the TRAIN split only
    # returns (fitted_model, extra_info_dict)

    pipe = build_pipeline(X_train)

    if not tune:
        # baseline settings
        pipe.set_params(
            model__n_estimators=500,
            model__max_depth=25,
            model__min_samples_leaf=5,
        )
        pipe.fit(X_train, y_train)
        return pipe, {"tuned": False}

    # hyperparameter search (CV inside)
    # ranges are good for a first pass without being absurdly slow
    param_distributions = {
        "model__n_estimators": [300, 500, 800, 1200],
        "model__max_depth": [None, 10, 15, 20, 25, 30, 40],
        "model__min_samples_leaf": [1, 2, 3, 5, 8, 10],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__max_features": ["sqrt", 0.5, 0.8, None],
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        # refit the best model on all X_train/y_train
        refit=True,
        verbose=0,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    extra = {
        "tuned": True,
        "cv_folds": cv_folds,
        # negative RMSE in log-space
        "cv_best_score_neg_rmse_log": float(search.best_score_),
        "best_params": dict(search.best_params_),
    }
    return best_model, extra


def save_model(model, path: str = "models/rf_model.pkl") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)