import os
import joblib
from ml_model.train import train_model

def test_train_model():
    train_model()
    assert os.path.exists("model.joblib"), "Model file not created"
    model = joblib.load("model.joblib")
    assert model is not None
