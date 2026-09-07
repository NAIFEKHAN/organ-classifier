"""
FastAPI backend for the organ classifier.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   -> basic status check
    POST /predict  -> upload an image, get back the predicted organ + confidence
"""

import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "organ_classifier.keras"
CONFIG_PATH = MODEL_DIR / "model_config.json"
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}

app = FastAPI(title="Organ Image Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a local demo; restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = []
img_size = 224  # overwritten from model_config.json at startup to match how the model was trained


@app.on_event("startup")
def load_model():
    global model, class_names, img_size
    if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
        # Server still starts so /health works, but /predict will report the issue clearly.
        print(f"WARNING: model not found in {MODEL_DIR}. Train the model first (see src/train.py).")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    class_names = config["class_names"]
    img_size = config["img_size"]
    print(f"Model loaded. img_size={img_size}, classes={class_names}")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((img_size, img_size))
    array = np.array(image, dtype=np.float32)
    array = tf.keras.applications.efficientnet.preprocess_input(array)
    return np.expand_dims(array, axis=0)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": class_names,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Train it first with src/train.py, "
                   "then restart the server.",
        )
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a PNG or JPEG image.")

    image_bytes = await file.read()
    try:
        input_array = preprocess_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this file as an image.")

    predictions = model.predict(input_array)[0]
    top_idx = int(np.argmax(predictions))
    confidence = float(predictions[top_idx]) * 100

    all_probs = sorted(
        [{"organ": class_names[i], "confidence": round(float(p) * 100, 2)}
         for i, p in enumerate(predictions)],
        key=lambda x: x["confidence"], reverse=True,
    )

    return {
        "prediction": class_names[top_idx],
        "confidence": round(confidence, 2),
        "all_probabilities": all_probs,
    }
