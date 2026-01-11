import os
import sys
import time
import threading
import subprocess
import requests
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException

# =========================
# GLOBAL STATE
# =========================

INSTALL_STATUS = {"stage": "waiting", "percent": 0}
MODEL_STATUS = {"stage": "waiting", "percent": 0}

MODEL_PATH = "models/arcface.onnx"
EMB_PATH = "embeddings.npy"
LBL_PATH = "labels.npy"

session = None
embeddings = []
labels = []

# =========================
# UTILS
# =========================

def log_status():
    print(
        f"[STATUS] install={INSTALL_STATUS} | model={MODEL_STATUS}",
        flush=True
    )

def every_5s_logger():
    while True:
        log_status()
        time.sleep(5)

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

# =========================
# MODEL DOWNLOAD
# =========================

def download_model():
    MODEL_STATUS["stage"] = "downloading"
    MODEL_STATUS["percent"] = 0

    safe_mkdir("models")

    url = (
        "https://github.com/onnx/models/raw/main/vision/body_analysis/"
        "arcface/model/arcfaceresnet100-8.onnx"
    )

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0

        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    MODEL_STATUS["percent"] = int(downloaded / total * 100)

    MODEL_STATUS["percent"] = 100

# =========================
# MODEL LOAD
# =========================

def load_model():
    global session

    MODEL_STATUS["stage"] = "loading"
    MODEL_STATUS["percent"] = 0

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    MODEL_STATUS["percent"] = 100
    MODEL_STATUS["stage"] = "ready"

# =========================
# EMBEDDING
# =========================

def get_embedding(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    emb = session.run(None, {"data": img})[0][0]
    emb = emb / np.linalg.norm(emb)
    return emb

# =========================
# STARTUP PIPELINE
# =========================

def startup_pipeline():
    try:
        # MODEL
        MODEL_STATUS["stage"] = "starting"
        MODEL_STATUS["percent"] = 0

        if not os.path.exists(MODEL_PATH):
            download_model()

        load_model()

        # LOAD DB
        global embeddings, labels
        if os.path.exists(EMB_PATH):
            embeddings = np.load(EMB_PATH).tolist()
            labels = np.load(LBL_PATH).tolist()

        print("[READY] System fully ready", flush=True)

    except Exception as e:
        print("[FATAL]", e, flush=True)
        sys.exit(1)

# =========================
# FASTAPI
# =========================

app = FastAPI()

@app.on_event("startup")
def startup():
    threading.Thread(target=every_5s_logger, daemon=True).start()
    threading.Thread(target=startup_pipeline, daemon=True).start()

@app.get("/health")
def health():
    return {
        "install_done": True,
        "model_ready": MODEL_STATUS["stage"] == "ready",
        "model_progress": MODEL_STATUS,
        "total_embeddings": len(embeddings)
    }

# =========================
# REGISTER
# =========================

@app.post("/register")
async def register(name: str, file: UploadFile = File(...)):
    if MODEL_STATUS["stage"] != "ready":
        return {"error": "Model not ready yet"}

    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "Invalid image")

    emb = get_embedding(img)

    embeddings.append(emb.tolist())
    labels.append(name)

    np.save(EMB_PATH, np.array(embeddings))
    np.save(LBL_PATH, np.array(labels))

    return {"status": "registered", "name": name}

# =========================
# IDENTIFY
# =========================

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    if MODEL_STATUS["stage"] != "ready":
        return {"error": "Model not ready yet"}

    if not embeddings:
        return {"error": "No registered faces"}

    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "Invalid image")

    emb = get_embedding(img)

    db = np.array(embeddings)
    sims = db @ emb
    idx = int(np.argmax(sims))
    score = float(sims[idx])

    return {
        "name": labels[idx],
        "confidence": score
    }
