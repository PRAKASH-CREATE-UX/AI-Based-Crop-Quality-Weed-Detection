# weed_app/utils/preprocessing.py

import cv2
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

# Resize and convert to grayscale
def preprocess_image(image_path):
    """
    Convert the input image to grayscale and resize to 224x224.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224))
    return resized

# Save the grayscale-resized image
def save_preprocessed_image(input_path, output_path):
    """
    Save the preprocessed grayscale image to output_path.
    """
    processed = preprocess_image(input_path)
    cv2.imwrite(output_path, processed)

# Background removal using DeepLabV3 (lightweight substitute for U^2-Net)
def remove_background(image_path, save_path):
    """
    Remove background from the image using DeepLabV3 segmentation.
    The object mask is applied to the original image and saved.
    """
    model = deeplabv3_resnet101(pretrained=True).eval()

    # Preprocessing
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()

    # Normalize mask to 0 or 255
    mask = np.where(mask == 15, 255, 0).astype(np.uint8)  # Class 15 is often "person", but acts as foreground

    # Resize image to match mask
    image_np = np.array(image.resize((224, 224)))
    if len(mask.shape) == 2:
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply mask to image
    result = cv2.bitwise_and(image_np, image_np, mask=mask)
    cv2.imwrite(save_path, result)
