import os
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# =======================
# BASE PATH
# =======================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

EMB_PATH = os.path.join(BASE_PATH, "embeddings.npy")
LBL_PATH = os.path.join(BASE_PATH, "labels.npy")
THR_PATH = os.path.join(BASE_PATH, "threshold.txt")

# =======================
# LOAD DATA
# =======================
embeddings = np.load(EMB_PATH)
labels = np.load(LBL_PATH)

with open(THR_PATH, "r") as f:
    AUTO_THRESHOLD = float(f.read())

# =======================
# FACE MODEL
# =======================
providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=providers
)
face_app.prepare(
    ctx_id=0 if torch.cuda.is_available() else -1,
    det_size=(640, 640)
)

# =======================
# FASTAPI
# =======================
app = FastAPI(
    title="Professional Face Recognition API",
    version="1.0.0"
)

# =======================
# HEALTH CHECK
# =======================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "threshold": AUTO_THRESHOLD,
        "total_embeddings": int(len(embeddings)),
        "unique_ids": int(len(set(labels.tolist())))
    }

# =======================
# IDENTIFY
# =======================
@app.post("/identify")
async def identify(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        img_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image file"}
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = face_app.get(img)
        if len(faces) != 1:
            return JSONResponse(
                status_code=400,
                content={"error": "Exactly one face must be visible"}
            )

        emb = faces[0].embedding.reshape(1, -1)
        sims = cosine_similarity(emb, embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_label = int(labels[best_idx])

        if best_score < AUTO_THRESHOLD:
            return {
                "recognized": False,
                "person_id": None,
                "score": round(best_score, 4)
            }

        return {
            "recognized": True,
            "person_id": best_label,
            "score": round(best_score, 4)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
