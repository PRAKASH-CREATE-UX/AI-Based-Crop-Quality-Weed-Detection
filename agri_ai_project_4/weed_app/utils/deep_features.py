# weed_app/utils/deep_features.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# Use ConvNeXt pretrained on ImageNet (use `weights=` API to avoid deprecation)
try:
    from torchvision.models import ConvNeXt_Base_Weights
    model = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
except Exception as e:
    cache_dir = None
    try:
        cache_dir = torch.hub.get_dir()
    except Exception:
        cache_dir = None
    msg = f"Failed to load ConvNeXt weights: {e}"
    if cache_dir:
        msg += f"\nIf you see a 'PytorchStreamReader failed reading zip archive' error, delete corrupted files in {os.path.join(cache_dir, 'checkpoints')} and retry."
    raise RuntimeError(msg)

model.classifier = torch.nn.Identity()  # Remove classification head
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_deep_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().numpy()
