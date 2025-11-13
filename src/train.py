import argparse
import mlflow
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import load_dataset

# MLflow running locally on port 5000
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("iris_poisoning")

def train(data_path):
    X_train, X_test, y_train, y_test = load_dataset(data_path)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc}")

    with mlflow.start_run():
        mlflow.log_param("data_path", data_path)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "model.pkl")
    print("Saved model → model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/iris_poisoned.csv")
    args = parser.parse_args()
    train(args.data)
