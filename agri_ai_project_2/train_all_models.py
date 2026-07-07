# train_all_models.py

from weed_app.utils.model_trainer import train_models
from weed_app.utils.train_deep_model import train_deep_rf

if __name__ == '__main__':
      
    train_models("datasets/cottonweed")            # Train ANN, RF, SVM on manual features
    train_deep_rf("datasets/cottonweed")           # Train Random Forest on ConvNeXt features
