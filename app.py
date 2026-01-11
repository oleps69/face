import os
import sys
import time
import threading
import traceback
import requests
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
from typing import Dict

# --------------------------
# CONFIG & GLOBAL VARIABLES
# --------------------------
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-11-int8.onnx"
MODEL_PATH = "/app/models/arcfaceresnet100-11-int8.onnx"
INSTALL_PROGRESS = {"stage": "waiting", "percent": 0}
MODEL_PROGRESS = {"stage": "waiting", "percent": 0}
TOTAL_EMBEDDINGS = 0
MODEL_SESSION: ort.InferenceSession | None = None

# --------------------------
# FASTAPI APP INIT
# --------------------------
app = FastAPI(title="Face Recognition API (INT8 Model)")

# --------------------------
# UTILITY FUNCTIONS
# --------------------------
def log_status():
    """Logs the current install/model status every 5 seconds."""
    while True:
        print(f"[STATUS] install={INSTALL_PROGRESS} | model={MODEL_PROGRESS}")
        time.sleep(5)

def download_model(url: str, save_path: str):
    """Downloads the ONNX model with progress logging."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    MODEL_PROGRESS["stage"] = "downloading"
    MODEL_PROGRESS["percent"] = 0

    resp = requests.get(url, stream=True)
    total_length = int(resp.headers.get('content-length', 0))
    chunk_size = 8192
    downloaded = 0

    with open(save_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                MODEL_PROGRESS["percent"] = int(downloaded / total_length * 100)

    MODEL_PROGRESS["percent"] = 100
    MODEL_PROGRESS["stage"] = "loaded"
    print(f"[MODEL] Download complete: {save_path}")

def load_model():
    """Loads the ONNX model into memory."""
    global MODEL_SESSION
    MODEL_PROGRESS["stage"] = "loading"
    MODEL_PROGRESS["percent"] = 0
    print("[MODEL] Loading ONNX model into memory...")
    MODEL_SESSION = ort.InferenceSession(MODEL_PATH)
    MODEL_PROGRESS["percent"] = 100
    MODEL_PROGRESS["stage"] = "ready"
    print("[MODEL] Model loaded successfully.")

def safe_numpy_from_file(file: UploadFile) -> np.ndarray:
    """Reads image file into numpy array, safe mode."""
    try:
        file_bytes = np.asarray(bytearray(file.file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image.")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

def compute_embedding(image: np.ndarray) -> np.ndarray:
    """Preprocess image and compute face embedding using the ONNX model."""
    if MODEL_SESSION is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    # Convert to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to 112x112
    img_resized = cv2.resize(img_rgb, (112, 112))
    # Normalize
    img_norm = (img_resized - 127.5) / 128.0
    img_input = np.transpose(img_norm, (2, 0, 1)).astype(np.float32)[np.newaxis, ...]
    inputs = {MODEL_SESSION.get_inputs()[0].name: img_input}
    embedding = MODEL_SESSION.run(None, inputs)[0]
    return embedding

# --------------------------
# BACKGROUND THREADS
# --------------------------
def startup_pipeline():
    """Handles model download and loading at startup."""
    global INSTALL_PROGRESS
    try:
        INSTALL_PROGRESS["stage"] = "installing"
        INSTALL_PROGRESS["percent"] = 0
        # Dummy install steps simulation
        for p in range(0, 101, 20):
            INSTALL_PROGRESS["percent"] = p
            print(f"[INSTALL] Installing dependencies... {p}%")
            time.sleep(1)
        INSTALL_PROGRESS["percent"] = 100
        INSTALL_PROGRESS["stage"] = "done"
        print("[INSTALL] Dependencies installed.")

        # Download and load model
        download_model(MODEL_URL, MODEL_PATH)
        load_model()
    except Exception as e:
        print("[FATAL] Startup pipeline failed!")
        traceback.print_exc()
        INSTALL_PROGRESS["stage"] = "error"
        MODEL_PROGRESS["stage"] = "error"

# --------------------------
# API ROUTES
# --------------------------
@app.get("/health")
def health_check():
    return {
        "install_done": INSTALL_PROGRESS["stage"] == "done",
        "model_ready": MODEL_PROGRESS["stage"] == "ready",
        "install_progress": INSTALL_PROGRESS,
        "model_progress": MODEL_PROGRESS,
        "total_embeddings": TOTAL_EMBEDDINGS
    }

@app.post("/identify")
def identify(file: UploadFile):
    if MODEL_SESSION is None or MODEL_PROGRESS["stage"] != "ready":
        return JSONResponse(status_code=503, content={"error": "Model not ready yet"})
    img = safe_numpy_from_file(file)
    embedding = compute_embedding(img)
    # Here you can add embedding database matching
    global TOTAL_EMBEDDINGS
    TOTAL_EMBEDDINGS += 1
    return {"embedding_shape": embedding.shape, "total_embeddings": TOTAL_EMBEDDINGS}

# --------------------------
# MAIN THREADS STARTUP
# --------------------------
if __name__ == "__main__":
    # Start status logging thread
    threading.Thread(target=log_status, daemon=True).start()
    # Start startup pipeline
    threading.Thread(target=startup_pipeline, daemon=True).start()
    # Run Uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
