from ml_model.train import train_model
from ml_model.evaluate import evaluate_model

def test_evaluate_model():
    train_model()
    try:
        evaluate_model()
    except Exception as e:
        assert False, f"Evaluation failed: {e}"
