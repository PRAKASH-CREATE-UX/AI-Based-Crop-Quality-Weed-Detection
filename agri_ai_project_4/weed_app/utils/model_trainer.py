import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from .feature_extraction import extract_features
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def train_models(dataset_folder):
    X, y = [], []
    for label_folder in os.listdir(dataset_folder):
        class_folder = os.path.join(dataset_folder, label_folder)
        if not os.path.isdir(class_folder):
            continue
        for img_file in os.listdir(class_folder):
            if img_file.endswith(".jpg") or img_file.endswith(".png"):
                path = os.path.join(class_folder, img_file)
                features = extract_features(path)
                X.append(features)
                y.append(label_folder)

    sm = SMOTE()
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

    # Create model folder if not exists
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Train models
    rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    svm = SVC(kernel='poly', degree=3, probability=True).fit(X_train, y_train)
    ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300).fit(X_train, y_train)

    # Save models
    joblib.dump(rf, os.path.join(model_dir, 'rf_model.pkl'))
    joblib.dump(svm, os.path.join(model_dir, 'svm_model.pkl'))
    joblib.dump(ann, os.path.join(model_dir, 'ann_model.pkl'))

    print("Manual feature models trained and saved successfully.")
