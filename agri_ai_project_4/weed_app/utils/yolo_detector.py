from ultralytics import YOLO
import os
from django.conf import settings
import glob

model = YOLO('yolov8m.pt')  # or yolov8n.pt

def run_yolo_detection(img_path):
    output_dir = os.path.join(settings.MEDIA_ROOT, 'yolo_output', 'predict')
    os.makedirs(output_dir, exist_ok=True)

    results = model(img_path, save=True, project=os.path.join(settings.MEDIA_ROOT, 'yolo_output'), name='predict', exist_ok=True)

    # Get latest output image path
    image_files = glob.glob(os.path.join(output_dir, '*.jpg'))
    image_files.sort(key=os.path.getmtime, reverse=True)
    if image_files:
        relative_path = os.path.relpath(image_files[0], settings.MEDIA_ROOT).replace("\\", "/")
        return relative_path
    else:
        return None
