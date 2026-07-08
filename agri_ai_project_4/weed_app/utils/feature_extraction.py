# weed_app/utils/feature_extraction.py
import cv2
import numpy as np
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from skimage.feature import local_binary_pattern

from scipy.stats import skew, kurtosis
from math import log2

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    features = []

    # GLCM Features
    glcm = greycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for p in props:
        features.append(greycoprops(glcm, p)[0][0])

    # Local Binary Pattern Histogram
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float") / (hist.sum() + 1e-7)
    features.extend(hist)

    # Statistical
    features.append(np.mean(image))
    features.append(np.std(image))
    features.append(skew(image.reshape(-1)))
    features.append(kurtosis(image.reshape(-1)))
    entropy = -np.sum([p * log2(p) for p in hist if p > 0])
    features.append(entropy)

    return features
