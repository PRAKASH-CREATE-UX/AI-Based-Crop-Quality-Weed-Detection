import joblib
import os
from .feature_extraction import extract_features

def predict_class(image_path):
    features = extract_features(image_path)

    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    rf_model = joblib.load(os.path.join(model_dir, 'rf_model.pkl'))
    svm_model = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
    ann_model = joblib.load(os.path.join(model_dir, 'ann_model.pkl'))

    rf_pred = rf_model.predict([features])[0]
    svm_pred = svm_model.predict([features])[0]
    ann_pred = ann_model.predict([features])[0]

    return {
        "Random Forest": rf_pred,
        "SVM": svm_pred,
        "ANN": ann_pred
    }
