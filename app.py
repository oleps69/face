""" Face Recognition API - Production Ready
    ArcFace ResNet100 INT8 ONNX Model
    Clean similarity scores (insightface-like range)
"""
import os
import sys
import time
import threading
import traceback
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json

# Third-party imports
import requests
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from pydantic import BaseModel

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
class Config:
    """Centralized configuration"""
    MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-11-int8.onnx"
    MODEL_NAME = "arcfaceresnet100-11-int8.onnx"
    MODEL_DIR = Path("/app/models")
    MODEL_PATH = MODEL_DIR / MODEL_NAME

    INPUT_SIZE = (112, 112)
    INPUT_MEAN = 127.5
    INPUT_STD = 128.0
    EMBEDDING_SIZE = 512

    HOST = "0.0.0.0"
    PORT = int(os.environ.get("PORT", 8080))
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    DOWNLOAD_CHUNK_SIZE = 8192
    STATUS_LOG_INTERVAL = 5
    REQUEST_TIMEOUT = 60

    STORAGE_DIR = Path("/app/storage")
    EMBEDDINGS_FILE = STORAGE_DIR / "embeddings.npy"
    LABELS_FILE = STORAGE_DIR / "labels.npy"
    THRESHOLD_FILE = STORAGE_DIR / "threshold.txt"
    LOGS_DIR = STORAGE_DIR / "logs"

    DEFAULT_THRESHOLD = 0.55      # insightface buffalo_l için yaygın iyi eşleşme eşiği

    @classmethod
    def setup_storage_paths(cls):
        possible_paths = [
            Path("/app/storage"),
            Path("/app"),
            Path("./storage"),
            Path(".")
        ]
        for path in possible_paths:
            emb_path = path / "embeddings.npy"
            if emb_path.exists():
                cls.STORAGE_DIR = path
                cls.EMBEDDINGS_FILE = emb_path
                cls.LABELS_FILE = path / "labels.npy"
                cls.THRESHOLD_FILE = path / "threshold.txt"
                cls.LOGS_DIR = path / "logs"
                return True
        # Default
        cls.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        return False


# ============================================================================
# GLOBAL STATE
# ============================================================================
class GlobalState:
    def __init__(self):
        self.lock = threading.Lock()
        self.model_session: Optional[ort.InferenceSession] = None
        self.embeddings_db: List[np.ndarray] = []     # normalized olacak!
        self.labels_db: List[str] = []
        self.total_requests = 0
        self.total_embeddings = 0
        self.default_threshold = Config.DEFAULT_THRESHOLD
        self.startup_time = time.time()
        self.errors: List[Dict] = []

    def set_model_session(self, session):
        with self.lock:
            self.model_session = session

    def get_model_session(self) -> Optional[ort.InferenceSession]:
        with self.lock:
            return self.model_session

    def increment_requests(self):
        with self.lock:
            self.total_requests += 1

    def add_error(self, error: Dict):
        with self.lock:
            self.errors.append(error)
            if len(self.errors) > 100:
                self.errors.pop(0)


state = GlobalState()


# ============================================================================
# LOGGING
# ============================================================================
class Logger:
    @staticmethod
    def setup():
        Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _log(level: str, message: str, data: Optional[Dict] = None):
        ts = datetime.now().isoformat()
        entry = {"timestamp": ts, "level": level, "message": message, "data": data or {}}
        print(f"[{ts}] [{level}] {message}", flush=True)
        if data:
            print(json.dumps(data, default=str), flush=True)

        try:
            log_file = Config.LOGS_DIR / f"app_{datetime.now():%Y%m%d}.log"
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except:
            pass

    @staticmethod
    def info(msg, data=None):    Logger._log("INFO", msg, data)
    @staticmethod
    def warning(msg, data=None): Logger._log("WARNING", msg, data)
    @staticmethod
    def error(msg, data=None):   Logger._log("ERROR", msg, data); state.add_error({"ts":datetime.now().isoformat(), "msg":msg, "data":data})
    @staticmethod
    def success(msg, data=None): Logger._log("SUCCESS", msg, data)


# ============================================================================
# MODEL MANAGER
# ============================================================================
class ModelManager:

    @staticmethod
    def download_model():
        if Config.MODEL_PATH.exists():
            Logger.info("Model zaten var", {"path": str(Config.MODEL_PATH)})
            return True

        Logger.info("Model indiriliyor...")
        try:
            r = requests.get(Config.MODEL_URL, stream=True, timeout=Config.REQUEST_TIMEOUT)
            r.raise_for_status()
            Config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(Config.MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=Config.DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            Logger.success("Model indirildi")
            return True
        except Exception as e:
            Logger.error("Model indirme başarısız", {"err": str(e)})
            return False

    @staticmethod
    def load_model() -> bool:
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)

            session = ort.InferenceSession(str(Config.MODEL_PATH), sess_options)
            state.set_model_session(session)
            Logger.success("Model yüklendi", {"providers": session.get_providers()})
            return True
        except Exception as e:
            Logger.error("Model yüklenemedi", {"err": str(e)})
            return False

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, Config.INPUT_SIZE)
        image = (image.astype(np.float32) - Config.INPUT_MEAN) / Config.INPUT_STD
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, 0)        # add batch
        return image

    @staticmethod
    def compute_embedding(image: np.ndarray) -> np.ndarray:
        session = state.get_model_session()
        if session is None:
            raise HTTPException(503, "Model henüz hazır değil")

        input_img = ModelManager.preprocess_image(image)
        input_name = session.get_inputs()[0].name

        t0 = time.time()
        embedding = session.run(None, {input_name: input_img})[0]
        t_inf = time.time() - t0

        # L2 NORMALIZE (önemli!)
        embedding = embedding / np.linalg.norm(embedding)

        Logger.debug("Embedding hesaplandı", {
            "inference_ms": round(t_inf * 1000, 1),
            "norm": float(np.linalg.norm(embedding))
        })

        return embedding.flatten()


# ============================================================================
# STORAGE
# ============================================================================
class Storage:

    @staticmethod
    def load_database():
        if not Config.EMBEDDINGS_FILE.exists():
            Logger.info("Veritabanı bulunamadı, sıfırdan başlıyor")
            return

        try:
            embeddings = np.load(Config.EMBEDDINGS_FILE)
            labels = np.load(Config.LABELS_FILE, allow_pickle=True).tolist()

            # ÖNEMLİ: Database'i normalize ediyoruz (bir kereye mahsus)
            norms = np.linalg.norm(embeddings, axis=1)
            if np.any(np.abs(norms - 1.0) > 0.05):  # kabaca kontrol
                Logger.info("Mevcut embedding'ler normalize ediliyor (ArcFace ONNX → insightface uyumu)")
                embeddings = embeddings / norms[:, np.newaxis]

            state.embeddings_db = list(embeddings)
            state.labels_db = labels
            state.total_embeddings = len(labels)

            if Config.THRESHOLD_FILE.exists():
                with open(Config.THRESHOLD_FILE) as f:
                    try:
                        state.default_threshold = float(f.read().strip())
                    except:
                        pass

            Logger.success("Veritabanı yüklendi", {
                "count": len(labels),
                "threshold": state.default_threshold,
                "normalized": True
            })

        except Exception as e:
            Logger.error("Veritabanı okuma hatası", {"err": str(e)})

    @staticmethod
    def save_database():
        try:
            if not state.embeddings_db:
                return

            emb_array = np.array(state.embeddings_db)
            np.save(Config.EMBEDDINGS_FILE, emb_array)
            np.save(Config.LABELS_FILE, np.array(state.labels_db, dtype=object))

            Logger.info("Veritabanı kaydedildi")
        except Exception as e:
            Logger.error("Veritabanı kaydetme hatası", {"err": str(e)})


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Face Recognition API",
    description="ArcFace ResNet100 - insightface tarzı temiz skorlar",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    Logger.setup()
    Config.setup_storage_paths()
    Storage.load_database()

    threading.Thread(target=lambda: (
        ModelManager.download_model() and
        ModelManager.load_model()
    ), daemon=True).start()

    threading.Thread(target=lambda: (
        time.sleep(4),
        Logger.info("Sistem durumu", state.get_status() if hasattr(state,'get_status') else {})
    ), daemon=True).start()


@app.get("/")
async def root():
    return {
        "service": "Face Recognition API",
        "status": "ok",
        "model": "ArcFace ResNet100 INT8",
        "threshold": state.default_threshold,
        "db_size": len(state.embeddings_db)
    }


@app.post("/register")
async def register(
    file: UploadFile = File(...),
    label: str = Form("unknown"),
    background_tasks: BackgroundTasks = None
):
    try:
        img = await file.read()
        nparr = np.frombuffer(img, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "Geçersiz resim")

        embedding = ModelManager.compute_embedding(image)

        state.embeddings_db.append(embedding)
        state.labels_db.append(label)
        state.total_embeddings += 1

        background_tasks.add_task(Storage.save_database)

        return {
            "success": True,
            "label": label,
            "db_size": len(state.labels_db)
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/search")
async def search(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None)
):
    if threshold is None:
        threshold = state.default_threshold

    try:
        img = await file.read()
        nparr = np.frombuffer(img, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "Geçersiz resim")

        query_emb = ModelManager.compute_embedding(image)

        if not state.embeddings_db:
            return {"success": True, "matches": [], "message": "Veritabanı boş"}

        # Cosine similarity (artık normalize oldukları için doğrudan dot product)
        similarities = []
        query_emb = query_emb.astype(np.float32)

        for i, db_emb in enumerate(state.embeddings_db):
            sim = float(np.dot(query_emb, db_emb))
            if sim >= threshold:
                similarities.append({
                    "index": i,
                    "label": state.labels_db[i],
                    "similarity": round(sim, 4),
                    "confidence": round(sim * 100, 2)
                })

        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "success": True,
            "matches": similarities[:15],  # en iyi 15
            "threshold_used": threshold,
            "total_checked": len(state.embeddings_db)
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/status")
async def status():
    return {
        "db_size": len(state.labels_db),
        "threshold": state.default_threshold,
        "model_ready": state.get_model_session() is not None,
        "uptime": int(time.time() - state.startup_time)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")
