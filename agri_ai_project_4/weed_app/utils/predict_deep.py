# weed_app/utils/predict_deep.py
import joblib
from .deep_features import extract_deep_features

def predict_deep(image_path):
    clf = joblib.load("weed_app/models/deep_rf.pkl")
    feat = extract_deep_features(image_path)
    return clf.predict([feat])[0]
