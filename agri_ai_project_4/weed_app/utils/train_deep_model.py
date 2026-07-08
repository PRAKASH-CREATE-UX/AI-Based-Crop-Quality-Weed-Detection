# weed_app/utils/train_deep_model.py
from .deep_features import extract_deep_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os, joblib
from imblearn.over_sampling import SMOTE

def train_deep_rf(dataset_dir):
    X, y = [], []
    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir): continue
        for file in os.listdir(class_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(class_dir, file)
                try:
                    feat = extract_deep_features(img_path)
                    X.append(feat)
                    y.append(label)
                except:
                    print(f"Skipping corrupt image: {img_path}")

    X, y = SMOTE().fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "weed_app/models/deep_rf.pkl")
    print("Deep feature RF model trained and saved.")
