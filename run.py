from src.features import load_data, clean_data, engineer_features, split_features_target
from src.train import train_model
from src.evaluate import evaluate

def main():
    df = load_data("data/kc_house_data.csv")
    df = clean_data(df)
    df = engineer_features(df)

    X, y = split_features_target(df)
    model, X_test, y_test = train_model(X, y)

    metrics = evaluate(model, X_test, y_test)
    print(metrics)

if __name__ == "__main__":
    main()