import os
import sys
import time
import threading
import subprocess
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# GLOBAL STATE
# =========================
INSTALL_DONE = False
MODEL_READY = False
INSTALL_PROGRESS = {"stage": "starting", "percent": 0}
MODEL_PROGRESS = {"stage": "waiting", "percent": 0}

EMBEDDINGS = None
LABELS = None
THRESHOLD = None
FACE_APP = None

BASE_DIR = os.path.dirname(__file__)

# =========================
# LOG THREAD (5 saniyede 1)
# =========================
def progress_logger():
    while not MODEL_READY:
        print(
            f"[STATUS] install={INSTALL_PROGRESS} | model={MODEL_PROGRESS}",
            flush=True
        )
        time.sleep(5)

# =========================
# LAZY INSTALL
# =========================
def lazy_install():
    global INSTALL_DONE

    packages = [
        "opencv-python-headless==4.9.0.80",
        "insightface==0.7.3",
        "onnxruntime==1.17.3"
    ]

    INSTALL_PROGRESS["stage"] = "installing packages"
    for i, pkg in enumerate(packages):
        INSTALL_PROGRESS["percent"] = int((i / len(packages)) * 100)
        print(f"[INSTALL] Installing {pkg}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

    INSTALL_PROGRESS["percent"] = 100
    INSTALL_PROGRESS["stage"] = "done"
    INSTALL_DONE = True

# =========================
# LOAD MODEL + DATA
# =========================
def load_model_and_data():
    global FACE_APP, EMBEDDINGS, LABELS, THRESHOLD, MODEL_READY

    from insightface.app import FaceAnalysis

    MODEL_PROGRESS["stage"] = "loading numpy files"
    MODEL_PROGRESS["percent"] = 10

    EMBEDDINGS = np.load(os.path.join(BASE_DIR, "embeddings.npy"))
    LABELS = np.load(os.path.join(BASE_DIR, "labels.npy"))

    with open(os.path.join(BASE_DIR, "threshold.txt")) as f:
        THRESHOLD = float(f.read().strip())

    MODEL_PROGRESS["stage"] = "initializing insightface"
    MODEL_PROGRESS["percent"] = 40

    FACE_APP = FaceAnalysis(providers=["CPUExecutionProvider"])
    FACE_APP.prepare(ctx_id=0)

    MODEL_PROGRESS["stage"] = "ready"
    MODEL_PROGRESS["percent"] = 100
    MODEL_READY = True

# =========================
# BACKGROUND STARTUP
# =========================
def startup_pipeline():
    lazy_install()
    load_model_and_data()

threading.Thread(target=progress_logger, daemon=True).start()
threading.Thread(target=startup_pipeline, daemon=True).start()

# =========================
# FASTAPI
# =========================
app = FastAPI(title="Face Recognition API")

@app.get("/health")
def health():
    return {
        "install_done": INSTALL_DONE,
        "model_ready": MODEL_READY,
        "install_progress": INSTALL_PROGRESS,
        "model_progress": MODEL_PROGRESS,
        "total_embeddings": 0 if EMBEDDINGS is None else len(EMBEDDINGS)
    }

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    if not MODEL_READY:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not ready yet"}
        )

    img_bytes = await file.read()
    img = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    faces = FACE_APP.get(img)
    if len(faces) != 1:
        return {"error": "Exactly one face required"}

    emb = faces[0].embedding.reshape(1, -1)
    sims = cosine_similarity(emb, EMBEDDINGS)[0]

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score < THRESHOLD:
        return {
            "result": "unknown",
            "score": round(best_score, 4)
        }

    return {
        "result": "recognized",
        "label": int(LABELS[best_idx]),
        "score": round(best_score, 4)
    }

# =========================
# LOCAL RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
