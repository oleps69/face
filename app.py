"""
Face Recognition API - Production Ready
Uses ArcFace ResNet100 INT8 ONNX Model
Complete implementation with comprehensive logging and error handling
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
    # Model Configuration
    MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-11-int8.onnx"
    MODEL_NAME = "arcfaceresnet100-11-int8.onnx"
    MODEL_DIR = Path("/app/models")
    MODEL_PATH = MODEL_DIR / MODEL_NAME
    
    # Input specifications for ArcFace ResNet100
    INPUT_SIZE = (112, 112)  # Height, Width
    INPUT_MEAN = 127.5
    INPUT_STD = 128.0
    EMBEDDING_SIZE = 512
    
    # API Configuration
    HOST = "0.0.0.0"
    PORT = int(os.environ.get("PORT", 8080))
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    # Performance Configuration
    DOWNLOAD_CHUNK_SIZE = 8192
    STATUS_LOG_INTERVAL = 5  # seconds
    REQUEST_TIMEOUT = 60  # seconds
    
    # Storage Configuration
    STORAGE_DIR = Path("/app/storage")
    EMBEDDINGS_FILE = STORAGE_DIR / "embeddings.npy"
    LABELS_FILE = STORAGE_DIR / "labels.json"
    LOGS_DIR = STORAGE_DIR / "logs"

# ============================================================================
# GLOBAL STATE MANAGEMENT
# ============================================================================

class GlobalState:
    """Thread-safe global state manager"""
    def __init__(self):
        self.lock = threading.Lock()
        self.install_progress = {"stage": "waiting", "percent": 0, "message": "Initializing..."}
        self.model_progress = {"stage": "waiting", "percent": 0, "message": "Waiting to start..."}
        self.model_session: Optional[ort.InferenceSession] = None
        self.total_embeddings = 0
        self.total_requests = 0
        self.embeddings_db: List[np.ndarray] = []
        self.labels_db: List[str] = []
        self.startup_time = time.time()
        self.errors: List[Dict] = []
        
    def update_install(self, stage: str, percent: int, message: str):
        with self.lock:
            self.install_progress = {"stage": stage, "percent": percent, "message": message}
            
    def update_model(self, stage: str, percent: int, message: str):
        with self.lock:
            self.model_progress = {"stage": stage, "percent": percent, "message": message}
            
    def set_model_session(self, session: ort.InferenceSession):
        with self.lock:
            self.model_session = session
            
    def get_model_session(self) -> Optional[ort.InferenceSession]:
        with self.lock:
            return self.model_session
            
    def increment_requests(self):
        with self.lock:
            self.total_requests += 1
            
    def increment_embeddings(self):
        with self.lock:
            self.total_embeddings += 1
            
    def add_error(self, error: Dict):
        with self.lock:
            self.errors.append(error)
            if len(self.errors) > 100:  # Keep last 100 errors
                self.errors.pop(0)
                
    def get_status(self) -> Dict:
        with self.lock:
            return {
                "install_progress": self.install_progress.copy(),
                "model_progress": self.model_progress.copy(),
                "total_embeddings": self.total_embeddings,
                "total_requests": self.total_requests,
                "uptime_seconds": int(time.time() - self.startup_time),
                "embeddings_in_db": len(self.embeddings_db),
                "model_ready": self.model_session is not None,
                "error_count": len(self.errors)
            }

# Initialize global state
state = GlobalState()

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class Logger:
    """Enhanced logging with file and console output"""
    
    @staticmethod
    def setup():
        """Create logs directory"""
        Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def _log(level: str, message: str, data: Optional[Dict] = None):
        """Internal logging method"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "data": data or {}
        }
        
        # Console output
        console_msg = f"[{timestamp}] [{level}] {message}"
        if data:
            console_msg += f" | {json.dumps(data, default=str)}"
        print(console_msg, flush=True)
        
        # File output
        try:
            log_file = Config.LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to write to log file: {e}", flush=True)
    
    @staticmethod
    def info(message: str, data: Optional[Dict] = None):
        Logger._log("INFO", message, data)
        
    @staticmethod
    def warning(message: str, data: Optional[Dict] = None):
        Logger._log("WARNING", message, data)
        
    @staticmethod
    def error(message: str, data: Optional[Dict] = None):
        Logger._log("ERROR", message, data)
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "data": data
        }
        state.add_error(error_entry)
        
    @staticmethod
    def success(message: str, data: Optional[Dict] = None):
        Logger._log("SUCCESS", message, data)
        
    @staticmethod
    def debug(message: str, data: Optional[Dict] = None):
        Logger._log("DEBUG", message, data)

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    """Handles model download, loading, and inference"""
    
    @staticmethod
    def download_with_progress(url: str, save_path: Path) -> bool:
        """Download model with progress tracking and verification"""
        try:
            Logger.info("Starting model download", {"url": url, "destination": str(save_path)})
            state.update_model("downloading", 0, "Initiating download...")
            
            # Create directory
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Start download with streaming
            response = requests.get(url, stream=True, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            Logger.info("Download started", {"total_size_mb": round(total_size / 1024 / 1024, 2)})
            
            downloaded = 0
            start_time = time.time()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=Config.DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress
                        if total_size > 0:
                            percent = int((downloaded / total_size) * 100)
                            speed_mbps = (downloaded / (1024 * 1024)) / max(time.time() - start_time, 0.1)
                            state.update_model(
                                "downloading", 
                                percent, 
                                f"Downloading... {percent}% ({speed_mbps:.2f} MB/s)"
                            )
            
            # Verify download
            actual_size = save_path.stat().st_size
            Logger.success("Model downloaded successfully", {
                "size_mb": round(actual_size / 1024 / 1024, 2),
                "time_seconds": round(time.time() - start_time, 2)
            })
            
            state.update_model("downloaded", 100, "Download complete, ready to load")
            return True
            
        except requests.exceptions.RequestException as e:
            Logger.error("Download failed - Network error", {"error": str(e), "traceback": traceback.format_exc()})
            state.update_model("error", 0, f"Download failed: {str(e)}")
            return False
        except Exception as e:
            Logger.error("Download failed - Unexpected error", {"error": str(e), "traceback": traceback.format_exc()})
            state.update_model("error", 0, f"Unexpected error: {str(e)}")
            return False
    
    @staticmethod
    def load_model(model_path: Path) -> Optional[ort.InferenceSession]:
        """Load ONNX model into memory"""
        try:
            Logger.info("Loading ONNX model", {"path": str(model_path)})
            state.update_model("loading", 0, "Loading model into memory...")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Configure ONNX Runtime session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = os.cpu_count() or 4
            sess_options.log_severity_level = 3  # Only errors
            
            Logger.debug("Creating ONNX Runtime session", {
                "threads": sess_options.intra_op_num_threads,
                "optimization": "ENABLE_ALL"
            })
            
            # Load model
            start_time = time.time()
            session = ort.InferenceSession(str(model_path), sess_options)
            load_time = time.time() - start_time
            
            # Get model info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            Logger.success("Model loaded successfully", {
                "load_time_seconds": round(load_time, 2),
                "input_name": input_info.name,
                "input_shape": input_info.shape,
                "output_name": output_info.name,
                "output_shape": output_info.shape,
                "providers": session.get_providers()
            })
            
            state.update_model("ready", 100, "Model ready for inference")
            return session
            
        except Exception as e:
            Logger.error("Model loading failed", {"error": str(e), "traceback": traceback.format_exc()})
            state.update_model("error", 0, f"Loading failed: {str(e)}")
            return None
    
    @staticmethod
    def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Preprocess image for ArcFace model with detailed logging"""
        try:
            original_shape = image.shape
            Logger.debug("Preprocessing image", {"original_shape": original_shape})
            
            # Convert BGR to RGB
            if len(image.shape) == 2:  # Grayscale
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:  # BGR
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to 112x112
            img_resized = cv2.resize(img_rgb, Config.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            
            # Normalize: (img - 127.5) / 128.0
            img_normalized = (img_resized.astype(np.float32) - Config.INPUT_MEAN) / Config.INPUT_STD
            
            # Transpose to CHW format (channels first)
            img_chw = np.transpose(img_normalized, (2, 0, 1))
            
            # Add batch dimension
            img_batch = np.expand_dims(img_chw, axis=0)
            
            metadata = {
                "original_shape": original_shape,
                "resized_shape": img_resized.shape,
                "final_shape": img_batch.shape,
                "dtype": str(img_batch.dtype),
                "value_range": [float(img_batch.min()), float(img_batch.max())]
            }
            
            Logger.debug("Image preprocessed", metadata)
            return img_batch, metadata
            
        except Exception as e:
            Logger.error("Image preprocessing failed", {"error": str(e), "traceback": traceback.format_exc()})
            raise
    
    @staticmethod
    def compute_embedding(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Compute face embedding with comprehensive error handling"""
        try:
            session = state.get_model_session()
            if session is None:
                raise HTTPException(status_code=503, detail="Model not ready")
            
            start_time = time.time()
            
            # Preprocess
            img_input, preprocess_meta = ModelManager.preprocess_image(image)
            preprocess_time = time.time() - start_time
            
            # Run inference
            inference_start = time.time()
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: img_input})
            inference_time = time.time() - inference_start
            
            # Extract embedding
            embedding = outputs[0]
            
            # Normalize embedding (L2 normalization)
            embedding_norm = embedding / np.linalg.norm(embedding)
            
            total_time = time.time() - start_time
            
            metadata = {
                "preprocess_time_ms": round(preprocess_time * 1000, 2),
                "inference_time_ms": round(inference_time * 1000, 2),
                "total_time_ms": round(total_time * 1000, 2),
                "embedding_shape": embedding_norm.shape,
                "embedding_norm": float(np.linalg.norm(embedding_norm))
            }
            
            Logger.debug("Embedding computed", metadata)
            return embedding_norm, metadata
            
        except Exception as e:
            Logger.error("Embedding computation failed", {"error": str(e), "traceback": traceback.format_exc()})
            raise

# ============================================================================
# FILE HANDLING
# ============================================================================

class FileHandler:
    """Handles file upload validation and processing"""
    
    @staticmethod
    def validate_file(file: UploadFile) -> Tuple[bool, Optional[str]]:
        """Validate uploaded file"""
        try:
            # Check filename
            if not file.filename:
                return False, "No filename provided"
            
            # Check extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in Config.ALLOWED_EXTENSIONS:
                return False, f"Invalid file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
            
            # Check content type
            content_type = file.content_type or ""
            if not content_type.startswith("image/"):
                return False, "File must be an image"
            
            return True, None
            
        except Exception as e:
            Logger.error("File validation error", {"error": str(e)})
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def read_image_from_upload(file: UploadFile) -> np.ndarray:
        """Read and decode image from upload"""
        try:
            Logger.debug("Reading uploaded file", {"filename": file.filename, "content_type": file.content_type})
            
            # Read file bytes
            file_bytes = file.file.read()
            file_size = len(file_bytes)
            
            # Check size
            if file_size > Config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size: {Config.MAX_FILE_SIZE / 1024 / 1024}MB"
                )
            
            Logger.debug("File read", {"size_kb": round(file_size / 1024, 2)})
            
            # Decode image
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(status_code=400, detail="Failed to decode image")
            
            Logger.debug("Image decoded", {"shape": img.shape, "dtype": str(img.dtype)})
            return img
            
        except HTTPException:
            raise
        except Exception as e:
            Logger.error("Failed to read image", {"error": str(e), "traceback": traceback.format_exc()})
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# ============================================================================
# STORAGE MANAGEMENT
# ============================================================================

class StorageManager:
    """Handles persistent storage of embeddings and labels"""
    
    @staticmethod
    def setup():
        """Initialize storage directories"""
        Config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        Logger.info("Storage initialized", {"path": str(Config.STORAGE_DIR)})
    
    @staticmethod
    def load_database():
        """Load embeddings and labels from disk"""
        try:
            if Config.EMBEDDINGS_FILE.exists() and Config.LABELS_FILE.exists():
                embeddings = np.load(Config.EMBEDDINGS_FILE)
                with open(Config.LABELS_FILE, 'r') as f:
                    labels = json.load(f)
                
                state.embeddings_db = list(embeddings)
                state.labels_db = labels
                state.total_embeddings = len(embeddings)
                
                Logger.success("Database loaded", {
                    "embeddings_count": len(embeddings),
                    "labels_count": len(labels)
                })
            else:
                Logger.info("No existing database found, starting fresh")
                
        except Exception as e:
            Logger.error("Failed to load database", {"error": str(e), "traceback": traceback.format_exc()})
    
    @staticmethod
    def save_database():
        """Save embeddings and labels to disk"""
        try:
            if state.embeddings_db:
                embeddings_array = np.array(state.embeddings_db)
                np.save(Config.EMBEDDINGS_FILE, embeddings_array)
                
                with open(Config.LABELS_FILE, 'w') as f:
                    json.dump(state.labels_db, f)
                
                Logger.success("Database saved", {
                    "embeddings_count": len(state.embeddings_db),
                    "labels_count": len(state.labels_db)
                })
        except Exception as e:
            Logger.error("Failed to save database", {"error": str(e), "traceback": traceback.format_exc()})

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

class BackgroundWorker:
    """Handles background threads"""
    
    @staticmethod
    def status_logger():
        """Periodically log system status"""
        Logger.info("Status logger started", {"interval_seconds": Config.STATUS_LOG_INTERVAL})
        
        while True:
            try:
                time.sleep(Config.STATUS_LOG_INTERVAL)
                status = state.get_status()
                Logger.info("System status", status)
            except Exception as e:
                Logger.error("Status logger error", {"error": str(e)})
    
    @staticmethod
    def startup_pipeline():
        """Main startup sequence"""
        try:
            Logger.info("=" * 80)
            Logger.info("STARTING FACE RECOGNITION API")
            Logger.info("=" * 80)
            
            # Step 1: Initialize storage
            state.update_install("initializing", 10, "Setting up storage...")
            Logger.setup()
            StorageManager.setup()
            time.sleep(0.5)
            
            # Step 2: Check Python environment
            state.update_install("checking", 20, "Checking Python environment...")
            Logger.info("Python environment", {
                "version": sys.version,
                "platform": sys.platform,
                "cpu_count": os.cpu_count()
            })
            time.sleep(0.5)
            
            # Step 3: Verify dependencies
            state.update_install("verifying", 40, "Verifying dependencies...")
            Logger.info("Dependencies verified", {
                "numpy": np.__version__,
                "cv2": cv2.__version__,
                "onnxruntime": ort.__version__
            })
            time.sleep(0.5)
            
            # Step 4: Load existing database
            state.update_install("loading_db", 60, "Loading existing database...")
            StorageManager.load_database()
            time.sleep(0.5)
            
            # Step 5: Complete installation
            state.update_install("done", 100, "Installation complete")
            Logger.success("Installation phase completed")
            time.sleep(0.5)
            
            # Step 6: Download model if needed
            if not Config.MODEL_PATH.exists():
                Logger.info("Model not found, starting download...")
                if not ModelManager.download_with_progress(Config.MODEL_URL, Config.MODEL_PATH):
                    raise Exception("Model download failed")
            else:
                Logger.info("Model already exists", {"path": str(Config.MODEL_PATH)})
                state.update_model("found", 100, "Model file found")
            
            # Step 7: Load model
            Logger.info("Loading model into memory...")
            session = ModelManager.load_model(Config.MODEL_PATH)
            
            if session is None:
                raise Exception("Model loading failed")
            
            state.set_model_session(session)
            
            # Step 8: Final checks
            Logger.success("=" * 80)
            Logger.success("FACE RECOGNITION API READY")
            Logger.success("=" * 80)
            Logger.info("Final status", state.get_status())
            
        except Exception as e:
            Logger.error("FATAL: Startup pipeline failed!", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            state.update_install("error", 0, f"Startup failed: {str(e)}")
            state.update_model("error", 0, f"Startup failed: {str(e)}")

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Create FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="Production-ready face recognition API using ArcFace ResNet100 INT8",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "Face Recognition API",
        "version": "2.0.0",
        "status": "operational",
        "model": "ArcFace ResNet100 INT8",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "identify": "/identify",
            "register": "/register",
            "search": "/search",
            "compare": "/compare",
            "database_stats": "/database/stats",
            "database_clear": "/database/clear",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    status = state.get_status()
    is_healthy = (
        status["install_progress"]["stage"] == "done" and
        status["model_progress"]["stage"] == "ready" and
        status["model_ready"]
    )
    
    return {
        "healthy": is_healthy,
        "timestamp": datetime.now().isoformat(),
        **status
    }

@app.get("/status")
def detailed_status():
    """Detailed status endpoint"""
    status = state.get_status()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "uptime_seconds": status["uptime_seconds"],
            "python_version": sys.version,
            "cpu_count": os.cpu_count()
        },
        "installation": status["install_progress"],
        "model": {
            "status": status["model_progress"],
            "ready": status["model_ready"],
            "path": str(Config.MODEL_PATH),
            "exists": Config.MODEL_PATH.exists()
        },
        "statistics": {
            "total_requests": status["total_requests"],
            "total_embeddings": status["total_embeddings"],
            "embeddings_in_database": status["embeddings_in_db"],
            "error_count": status["error_count"]
        },
        "recent_errors": state.errors[-5:] if state.errors else []
    }

@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    """
    Identify a face from uploaded image
    Returns embedding vector and metadata
    """
    request_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
    Logger.info(f"[{request_id}] Identify request received", {"filename": file.filename})
    
    try:
        state.increment_requests()
        
        # Validate model is ready
        if state.get_model_session() is None:
            Logger.warning(f"[{request_id}] Model not ready")
            raise HTTPException(status_code=503, detail="Model not ready yet. Please wait.")
        
        # Validate file
        is_valid, error_msg = FileHandler.validate_file(file)
        if not is_valid:
            Logger.warning(f"[{request_id}] Invalid file", {"error": error_msg})
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Read and decode image
        image = FileHandler.read_image_from_upload(file)
        
        # Compute embedding
        embedding, metadata = ModelManager.compute_embedding(image)
        
        # Increment counter
        state.increment_embeddings()
        
        Logger.success(f"[{request_id}] Embedding computed successfully")
        
        return {
            "success": True,
            "request_id": request_id,
            "embedding": embedding.flatten().tolist(),
            "embedding_shape": list(embedding.shape),
            "embedding_size": int(np.prod(embedding.shape)),
            "metadata": metadata,
            "statistics": {
                "total_embeddings_computed": state.total_embeddings,
                "total_requests": state.total_requests
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        Logger.error(f"[{request_id}] Identify request failed", {
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/register")
async def register_face(
    file: UploadFile = File(...),
    label: str = Form("unknown"),
    background_tasks: BackgroundTasks = None
):
    """
    Register a face in the database
    Computes embedding and stores it with a label
    """
    request_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
    Logger.info(f"[{request_id}] Register request", {"label": label, "filename": file.filename})
    
    try:
        # Validate model
        if state.get_model_session() is None:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        # Validate file
        is_valid, error_msg = FileHandler.validate_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Read image
        image = FileHandler.read_image_from_upload(file)
        
        # Compute embedding
        embedding, metadata = ModelManager.compute_embedding(image)
        
        # Add to database
        state.embeddings_db.append(embedding.flatten())
        state.labels_db.append(label)
        state.increment_embeddings()
        
        # Save in background
        if background_tasks:
            background_tasks.add_task(StorageManager.save_database)
        else:
            StorageManager.save_database()
        
        Logger.success(f"[{request_id}] Face registered", {"label": label})
        
        return {
            "success": True,
            "request_id": request_id,
            "label": label,
            "database_size": len(state.embeddings_db),
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        Logger.error(f"[{request_id}] Register failed", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_face(file: UploadFile = File(...), threshold: float = Form(0.6)):
    """
    Search for a face in the database
    Returns best matches above threshold
    """
    request_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
    Logger.info(f"[{request_id}] Search request", {"threshold": threshold})
    
    try:
        # Validate
        if state.get_model_session() is None:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        if not state.embeddings_db:
            return {
                "success": True,
                "request_id": request_id,
                "matches": [],
                "message": "Database is empty"
            }
        
        # Validate file
        is_valid, error_msg = FileHandler.validate_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Read image
        image = FileHandler.read_image_from_upload(file)
        
        # Compute embedding
        query_embedding, metadata = ModelManager.compute_embedding(image)
        query_flat = query_embedding.flatten()
        
        # Search in database
        similarities = []
        for idx, db_embedding in enumerate(state.embeddings_db):
            # Cosine similarity
            similarity = float(np.dot(query_flat, db_embedding))
            if similarity >= threshold:
                similarities.append({
                    "index": idx,
                    "label": state.labels_db[idx],
                    "similarity": round(similarity, 4),
                    "confidence": round(similarity * 100, 2)
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        Logger.success(f"[{request_id}] Search completed", {
            "matches_found": len(similarities),
            "database_size": len(state.embeddings_db)
        })
        
        return {
            "success": True,
            "request_id": request_id,
            "matches": similarities,
            "total_checked": len(state.embeddings_db),
            "threshold": threshold,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        Logger.error(f"[{request_id}] Search failed", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Compare two faces and return similarity score
    """
    request_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
    Logger.info(f"[{request_id}] Compare request", {
        "file1": file1.filename,
        "file2": file2.filename
    })
    
    try:
        # Validate model
        if state.get_model_session() is None:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        # Validate files
        is_valid1, error_msg1 = FileHandler.validate_file(file1)
        is_valid2, error_msg2 = FileHandler.validate_file(file2)
        
        if not is_valid1:
            raise HTTPException(status_code=400, detail=f"File 1: {error_msg1}")
        if not is_valid2:
            raise HTTPException(status_code=400, detail=f"File 2: {error_msg2}")
        
        # Read images
        image1 = FileHandler.read_image_from_upload(file1)
        image2 = FileHandler.read_image_from_upload(file2)
        
        # Compute embeddings
        embedding1, metadata1 = ModelManager.compute_embedding(image1)
        embedding2, metadata2 = ModelManager.compute_embedding(image2)
        
        # Calculate similarity (cosine similarity)
        similarity = float(np.dot(embedding1.flatten(), embedding2.flatten()))
        
        # Determine match
        is_match = similarity >= 0.6  # Default threshold
        confidence = round(similarity * 100, 2)
        
        Logger.success(f"[{request_id}] Comparison completed", {
            "similarity": similarity,
            "is_match": is_match
        })
        
        return {
            "success": True,
            "request_id": request_id,
            "similarity": round(similarity, 4),
            "confidence": confidence,
            "is_match": is_match,
            "metadata": {
                "file1": metadata1,
                "file2": metadata2
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        Logger.error(f"[{request_id}] Compare failed", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/stats")
def database_stats():
    """Get database statistics"""
    try:
        label_counts = {}
        for label in state.labels_db:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            "success": True,
            "total_embeddings": len(state.embeddings_db),
            "total_labels": len(state.labels_db),
            "unique_labels": len(set(state.labels_db)) if state.labels_db else 0,
            "label_distribution": dict(
                sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            ),
            "storage": {
                "embeddings_file": str(Config.EMBEDDINGS_FILE),
                "labels_file": str(Config.LABELS_FILE),
                "embeddings_exists": Config.EMBEDDINGS_FILE.exists(),
                "labels_exists": Config.LABELS_FILE.exists()
            }
        }
    except Exception as e:
        Logger.error("Failed to get database stats", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/database/clear")
def clear_database():
    """Clear entire database"""
    try:
        Logger.warning("Database clear requested")
        
        # Clear in-memory data
        state.embeddings_db.clear()
        state.labels_db.clear()
        state.total_embeddings = 0
        
        # Remove files
        if Config.EMBEDDINGS_FILE.exists():
            Config.EMBEDDINGS_FILE.unlink()
            Logger.info("Embeddings file deleted")
        
        if Config.LABELS_FILE.exists():
            Config.LABELS_FILE.unlink()
            Logger.info("Labels file deleted")
        
        Logger.success("Database cleared successfully")
        
        return {
            "success": True,
            "message": "Database cleared successfully",
            "embeddings_removed": True,
            "labels_removed": True
        }
        
    except Exception as e:
        Logger.error("Failed to clear database", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/database/remove/{index}")
def remove_embedding(index: int):
    """Remove specific embedding by index"""
    try:
        if index < 0 or index >= len(state.embeddings_db):
            raise HTTPException(status_code=404, detail=f"Index {index} not found")
        
        removed_label = state.labels_db[index]
        
        # Remove from database
        del state.embeddings_db[index]
        del state.labels_db[index]
        
        # Save database
        StorageManager.save_database()
        
        Logger.info(f"Removed embedding at index {index}", {"label": removed_label})
        
        return {
            "success": True,
            "message": f"Removed embedding at index {index}",
            "removed_label": removed_label,
            "new_database_size": len(state.embeddings_db)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        Logger.error("Failed to remove embedding", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs/recent")
def recent_logs(count: int = 20):
    """Get recent log entries"""
    try:
        log_file = Config.LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        
        if not log_file.exists():
            return {
                "success": True,
                "logs": [],
                "message": "No logs found for today"
            }
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Get last N lines
        recent = lines[-count:] if len(lines) > count else lines
        logs = []
        
        for line in recent:
            if line.strip():
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        return {
            "success": True,
            "logs": logs,
            "count": len(logs),
            "total_lines": len(lines)
        }
        
    except Exception as e:
        Logger.error("Failed to read logs", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/errors/recent")
def recent_errors(count: int = 10):
    """Get recent errors"""
    try:
        errors = state.errors[-count:] if state.errors else []
        return {
            "success": True,
            "errors": errors,
            "count": len(errors),
            "total_errors": len(state.errors)
        }
    except Exception as e:
        Logger.error("Failed to get errors", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/reload")
def reload_model():
    """Reload the model (useful after update)"""
    try:
        Logger.info("Manual model reload requested")
        
        if not Config.MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Load model
        session = ModelManager.load_model(Config.MODEL_PATH)
        
        if session is None:
            raise HTTPException(status_code=500, detail="Failed to load model")
        
        # Update global state
        state.set_model_session(session)
        
        Logger.success("Model reloaded successfully")
        
        return {
            "success": True,
            "message": "Model reloaded successfully",
            "model_info": {
                "path": str(Config.MODEL_PATH),
                "size_mb": round(Config.MODEL_PATH.stat().st_size / 1024 / 1024, 2),
                "exists": Config.MODEL_PATH.exists()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        Logger.error("Model reload failed", {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
def get_config():
    """Get current configuration"""
    return {
        "success": True,
        "model": {
            "url": Config.MODEL_URL,
            "name": Config.MODEL_NAME,
            "path": str(Config.MODEL_PATH),
            "input_size": Config.INPUT_SIZE,
            "embedding_size": Config.EMBEDDING_SIZE,
            "input_mean": Config.INPUT_MEAN,
            "input_std": Config.INPUT_STD
        },
        "api": {
            "host": Config.HOST,
            "port": Config.PORT,
            "max_file_size_mb": round(Config.MAX_FILE_SIZE / 1024 / 1024, 2),
            "allowed_extensions": list(Config.ALLOWED_EXTENSIONS),
            "request_timeout": Config.REQUEST_TIMEOUT
        },
        "storage": {
            "storage_dir": str(Config.STORAGE_DIR),
            "embeddings_file": str(Config.EMBEDDINGS_FILE),
            "labels_file": str(Config.LABELS_FILE),
            "logs_dir": str(Config.LOGS_DIR)
        },
        "performance": {
            "download_chunk_size": Config.DOWNLOAD_CHUNK_SIZE,
            "status_log_interval": Config.STATUS_LOG_INTERVAL
        }
    }

# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """FastAPI startup event"""
    Logger.info("FastAPI application starting...")
    
    # Start background threads
    threading.Thread(
        target=BackgroundWorker.status_logger,
        daemon=True,
        name="StatusLogger"
    ).start()
    
    threading.Thread(
        target=BackgroundWorker.startup_pipeline,
        daemon=True,
        name="StartupPipeline"
    ).start()
    
    Logger.info("Background workers started")

@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI shutdown event"""
    Logger.info("FastAPI application shutting down...")
    
    # Save database before shutdown
    try:
        StorageManager.save_database()
        Logger.success("Database saved on shutdown")
    except Exception as e:
        Logger.error("Failed to save database on shutdown", {"error": str(e)})
    
    Logger.info("Shutdown complete")

# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    Logger.warning("HTTP Exception", {
        "status_code": exc.status_code,
        "detail": exc.detail,
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    Logger.error("Unhandled exception", {
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    print("=" * 80)
    print("FACE RECOGNITION API - PRODUCTION READY")
    print("=" * 80)
    print(f"Model: ArcFace ResNet100 INT8")
    print(f"Version: 2.0.0")
    print(f"Host: {Config.HOST}")
    print(f"Port: {Config.PORT}")
    print(f"Docs: http://{Config.HOST}:{Config.PORT}/docs")
    print(f"Storage: {Config.STORAGE_DIR}")
    print("=" * 80)
    print()
    
    import uvicorn
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
        limit_concurrency=100,
        limit_max_requests=10000
    )

if __name__ == "__main__":
    main()
