import os
import sys
import time
import threading
import subprocess
from typing import Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# ==============================
# GLOBAL STATE
# ==============================
install_progress = {"stage": "waiting", "percent": 0}
model_progress = {"stage": "waiting", "percent": 0}

install_done = False
model_ready = False

embeddings = None
labels = None
threshold = None
persons = None
face_app = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# HELPERS
# ==============================
def log(msg: str):
    print(msg, flush=True)

def pip_install(pkg: str, percent_from: int, percent_to: int):
    global install_progress
    install_progress["stage"] = "installing"

    log(f"[INSTALL] {pkg}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            pkg,
            "--only-binary=:all:",
            "--no-cache-dir",
        ]
    )
    install_progress["percent"] = percent_to

# ==============================
# STARTUP PIPELINE
# ==============================
def startup_pipeline():
    global install_done, model_ready
    global embeddings, labels, threshold, persons, face_app

    try:
        # ---------- INSTALL ----------
        install_progress["stage"] = "starting"
        install_progress["percent"] = 0

        pip_install("scikit-learn", 0, 20)
        pip_install("scipy", 20, 35)
        pip_install("onnxruntime", 35, 50)
        pip_install("torch", 50, 65)
        pip_install("insightface", 65, 80)

        install_progress["percent"] = 100
        install_progress["stage"] = "done"
        install_done = True

        # ---------- MODEL LOAD ----------
        model_progress["stage"] = "loading"
        model_progress["percent"] = 0

        import numpy as np
        import cv2
        from sklearn.metrics.pairwise import cosine_similarity
        from insightface.app import FaceAnalysis

        model_progress["percent"] = 30

        # Load data
        embeddings = np.load(os.path.join(BASE_DIR, "embeddings.npy"))
        labels = np.load(os.path.join(BASE_DIR, "labels.npy"))

        with open(os.path.join(BASE_DIR, "threshold.txt")) as f:
            threshold = float(f.read().strip())

        # Optional persons.txt
        persons_path = os.path.join(BASE_DIR, "persons.txt")
        if os.path.exists(persons_path):
            with open(persons_path) as f:
                persons = [x.strip() for x in f.readlines()]
        else:
            persons = [f"person_{i}" for i in range(len(set(labels)))]

        model_progress["percent"] = 60

        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        face_app.prepare(ctx_id=0, det_size=(640, 640))

        model_progress["percent"] = 100
        model_progress["stage"] = "ready"
        model_ready = True

        log("[SYSTEM] Model ready")

    except Exception as e:
        log(f"[FATAL] {e}")
        install_progress["stage"] = "error"
        model_progress["stage"] = "error"

# ==============================
# FASTAPI
# ==============================
app = FastAPI(title="Face Recognition API")

@app.on_event("startup")
def on_startup():
    threading.Thread(target=startup_pipeline, daemon=True).start()

@app.get("/health")
def health():
    return {
        "install_done": install_done,
        "model_ready": model_ready,
        "install_progress": install_progress,
        "model_progress": model_progress,
        "total_embeddings": 0 if embeddings is None else len(embeddings),
    }

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    if not model_ready:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not ready yet"}
        )

    import numpy as np
    import cv2
    from sklearn.metrics.pairwise import cosine_similarity

    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    if len(faces) != 1:
        return {"error": "Exactly one face required"}

    emb = faces[0].embedding.reshape(1, -1)
    sims = cosine_similarity(emb, embeddings)[0]

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score < threshold:
        return {
            "match": False,
            "score": round(best_score, 4),
        }

    return {
        "match": True,
        "person": persons[labels[best_idx]],
        "score": round(best_score, 4),
    }
