import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate_model():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    model = joblib.load("model.joblib")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {acc:.2f}")

if __name__ == "__main__":
    evaluate_model()
