from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load model once
model = YOLO("best.pt")


def predict_image(image, conf_threshold=0.25):
    """
    Run YOLO prediction on a PIL image.
    Returns:
        annotated_image: numpy array with boxes drawn
        predictions: list of dicts with class and confidence
    """
    results = model.predict(image, conf=conf_threshold, verbose=False)
    result = results[0]

    annotated_image = result.plot()

    predictions = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = model.names[class_id]

            predictions.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4)
            })

    return annotated_image, predictions