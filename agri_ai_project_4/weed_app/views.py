# weed_app/views.py

from django.shortcuts import render
from .forms import ImageUploadForm
import os
from django.conf import settings
from django.contrib.auth.decorators import login_required
from weed_app.utils.preprocessing import save_preprocessed_image, remove_background
from weed_app.utils.predictor import predict_class
from weed_app.utils.predict_deep import predict_deep
from weed_app.utils.yolo_detector import run_yolo_detection
from collections import Counter

MEDIA_URL = settings.MEDIA_URL


@login_required
def upload_image(request):
    context = {}
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data['image']
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')
            os.makedirs(upload_dir, exist_ok=True)

            # Save uploaded image
            original_path = os.path.join(upload_dir, image_file.name)
            with open(original_path, 'wb+') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            # Preprocessing
            gray_path = original_path.replace('.jpg', '_gray.jpg').replace('.png', '_gray.png')
            bg_path = original_path.replace('.jpg', '_bg.jpg').replace('.png', '_bg.png')

            save_preprocessed_image(original_path, gray_path)
            remove_background(original_path, bg_path)

            # Manual Feature Predictions
            predictions = predict_class(gray_path)

            # Deep Feature Prediction
            deep_pred = predict_deep(original_path)

            # Combined Prediction Voting
            all_preds = list(predictions.values()) + [deep_pred]
            final_pred = Counter(all_preds).most_common(1)[0][0]

            # YOLOv8 Detection
            yolo_output_rel_path = run_yolo_detection(original_path)
            yolo_output_link = f"{MEDIA_URL}{yolo_output_rel_path.replace(os.sep, '/')}"

            # Prepare context for HTML
            context.update({
                'form': form,
                'original_img': f"uploaded_images/{os.path.basename(original_path)}",
                'gray_img': f"uploaded_images/{os.path.basename(gray_path)}",
                'predictions': predictions,
                'deep_pred': deep_pred,
                'final_pred': final_pred,
                'yolo_output': yolo_output_link,  # ✅ fixed URL
            })
    else:
        form = ImageUploadForm()
        context['form'] = form

    return render(request, 'upload_image.html', context)
