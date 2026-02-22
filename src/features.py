# turning the raw dataframe into clean matrix
import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df[(df["bedrooms"] > 0) & (df["bathrooms"] > 0)]
    return df

def engineer_features(df):
    d = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
    df = df.copy()
    df["sale_year"] = d.dt.year
    df["sale_month"] = d.dt.month
    return df

def split_features_target(df):
    X = df.drop(columns=["price", "date", "id"], errors="ignore")
    y = np.log1p(df["price"])
    return X, y