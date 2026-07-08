#!/usr/bin/env python
"""Retrain deep RF emphasizing wrong predictions.

Usage:
    python retrain_on_errors.py datasets/cottonweed --emphasize 5

This will scan the dataset folder (class subfolders), run current model (if present)
to list wrong predictions, then retrain a RandomForest where misclassified examples
are duplicated `--emphasize` times to give them more weight.
"""
import os
import sys
import joblib
from .deep_features import extract_deep_features
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import argparse


def collect_features(dataset_dir):
    X, y, paths = [], [], []
    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for file in os.listdir(class_dir):
            if not (file.lower().endswith('.jpg') or file.lower().endswith('.png')):
                continue
            img_path = os.path.join(class_dir, file)
            try:
                feat = extract_deep_features(img_path)
                X.append(feat)
                y.append(label)
                paths.append(img_path)
            except Exception as e:
                print(f"Skipping corrupt image: {img_path} ({e})")
    return X, y, paths


def list_misclassifications(model, X, y, paths):
    wrong = []
    if model is None:
        return wrong
    preds = model.predict(X)
    for p, t, path in zip(preds, y, paths):
        if str(p) != str(t):
            wrong.append((path, t, p))
    return wrong


def retrain(dataset_dir, emphasize=1, out_path="weed_app/models/deep_rf.pkl"):
    X, y, paths = collect_features(dataset_dir)
    model = None
    try:
        model = joblib.load(out_path)
        print(f"Loaded existing model from {out_path}")
    except Exception:
        print("No existing model found — a fresh model will be trained.")

    wrong = list_misclassifications(model, X, y, paths)
    if wrong:
        print("Misclassified examples:")
        for path, true, pred in wrong:
            print(f"  {path}  -> predicted: {pred}  true: {true}")
    else:
        print("No existing model or no misclassifications found.")

    # Emphasize wrong examples by duplicating them
    X_train, y_train = list(X), list(y)
    if emphasize > 1 and wrong:
        wrong_set = set(w[0] for w in wrong)
        for i, path in enumerate(paths):
            if path in wrong_set:
                for _ in range(emphasize - 1):
                    X_train.append(X[i])
                    y_train.append(y[i])

    # Optional SMOTE to balance
    try:
        X_res, y_res = SMOTE().fit_resample(X_train, y_train)
    except Exception:
        X_res, y_res = X_train, y_train

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_res, y_res)
    joblib.dump(clf, out_path)
    print(f"Retrained model saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    parser.add_argument('--emphasize', type=int, default=1,
                        help='How many times to duplicate misclassified examples (default 1)')
    args = parser.parse_args()
    if not os.path.isdir(args.dataset_dir):
        print(f"Dataset directory not found: {args.dataset_dir}")
        sys.exit(2)
    retrain(args.dataset_dir, emphasize=args.emphasize)


if __name__ == '__main__':
    main()
