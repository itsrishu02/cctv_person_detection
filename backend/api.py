"""
ZEEX AI - FastAPI backend for the Vercel/v0 interface.

Runs the existing OpenCV recognition pipeline locally and exposes a small HTTP
API for the Next.js frontend. The long-running camera/video loop stays in
Python because Vercel UI processes are not a good place for YOLO/OpenCV stream
inference.
"""
from __future__ import annotations

import os

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|buffer_size;1048576|reconnect;1|reconnect_streamed;1|reconnect_delay_max;2",
)

import shutil
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from face_pipeline import load_config, load_known_faces
from recognizer import FrameRecognizer


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class StartRequest(BaseModel):
    source: str
    cameraId: str = "CAM-01"
    zone: str = "Zone-A"
    threshold: Optional[float] = None
    detectEvery: Optional[int] = None
    faceImgSize: Optional[int] = None
    faceConf: Optional[float] = None
    minFace: Optional[int] = None
    emptyZoneFrames: Optional[int] = None
    personEnabled: Optional[bool] = None
    personConf: Optional[float] = None


class SettingsRequest(BaseModel):
    cameraId: Optional[str] = None
    zone: Optional[str] = None
    threshold: Optional[float] = None
    detectEvery: Optional[int] = None
    faceImgSize: Optional[int] = None
    faceConf: Optional[float] = None
    minFace: Optional[int] = None
    emptyZoneFrames: Optional[int] = None
    personEnabled: Optional[bool] = None
    personConf: Optional[float] = None


def default_stats() -> dict:
    return {
        "fps": 0.0,
        "detections": 0,
        "known_count": 0,
        "unknown_count": 0,
        "person_no_face_count": 0,
        "total_people_count": 0,
        "no_person_frame_count": 0,
        "empty_zone_threshold": 3,
        "empty_zone": False,
        "zone_status": "IDLE",
    }


class RecognitionState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.recognizer: Optional[FrameRecognizer] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.run_id = ""
        self.running = False
        self.message = "Idle"
        self.latest_jpeg: Optional[bytes] = None
        self.latest_stats = default_stats()


state = RecognitionState()
app = FastAPI(title="ZEEX Face Recognition v0 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def open_capture(source: str) -> cv2.VideoCapture:
    resolved_source: str | int = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(resolved_source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(resolved_source)
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap


def source_open_error(source: str) -> str:
    if source.startswith("rtsp://"):
        return (
            "Could not open RTSP source. Check the camera IP, port, stream path, "
            f"credentials, and whether VLC can open this exact URL: {source}"
        )
    if source.startswith(("http://", "https://")):
        return (
            "Could not open network video source. Check that the URL returns a "
            f"playable video stream: {source}"
        )
    if source.isdigit():
        return f"Could not open camera index {source}."
    return (
        "Could not open local video file. Check that the backend machine can "
        f"access this path: {source}"
    )


def get_recognizer() -> FrameRecognizer:
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=500, detail="config.yaml not found")
    encodings_path = BASE_DIR / "encodings.pkl"
    if not encodings_path.exists():
        raise HTTPException(
            status_code=500,
            detail="encodings.pkl not found. Run python encode_faces.py first.",
        )
    with state.lock:
        if state.recognizer is None:
            cfg = load_config(str(CONFIG_PATH))
            state.recognizer = FrameRecognizer(cfg, BASE_DIR)
        return state.recognizer


def reset_recognizer_runtime(recognizer: FrameRecognizer) -> None:
    recognizer.frame_idx = 0
    recognizer.tracked = []
    recognizer.tracked_persons = []
    recognizer._no_person_frame_count = 0
    recognizer._fps = 0.0
    recognizer._t_prev = time.time()


def apply_settings(
    recognizer: FrameRecognizer,
    req: StartRequest | SettingsRequest,
) -> None:
    camera_id = getattr(req, "cameraId", None)
    zone = getattr(req, "zone", None)
    if camera_id is not None or zone is not None:
        recognizer.update_camera_zone(camera_id, zone)
    if req.threshold is not None:
        recognizer.threshold = float(req.threshold)
    if req.detectEvery is not None:
        recognizer.detect_every_n = max(1, int(req.detectEvery))
    if req.faceImgSize is not None:
        recognizer.detector.imgsz = int(req.faceImgSize)
    if req.faceConf is not None:
        recognizer.detector.conf = float(req.faceConf)
    if req.minFace is not None:
        recognizer.min_face_for_embed = int(req.minFace)
    if req.emptyZoneFrames is not None:
        recognizer.empty_zone_threshold_frames = max(1, int(req.emptyZoneFrames))
    if req.personEnabled is not None:
        recognizer.person_enabled = bool(req.personEnabled)
    if req.personConf is not None and recognizer.person_detector is not None:
        recognizer.person_detector.conf = float(req.personConf)


def stop_current(wait: bool = True) -> None:
    with state.lock:
        state.stop_event.set()
        thread = state.thread
        state.running = False
        state.message = "Stopping"
    if wait and thread is not None and thread.is_alive():
        thread.join(timeout=5.0)


class LatestFrameSlot:
    """Holds the most recent decoded frame, overwriting older ones."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0

    def put(self, frame) -> None:
        with self._lock:
            self._frame = frame
            self._frame_id += 1

    def get_if_new(self, last_id: int):
        with self._lock:
            if self._frame is None or self._frame_id == last_id:
                return None, last_id
            return self._frame, self._frame_id


def recognition_loop(
    source: str,
    recognizer: FrameRecognizer,
    stop_event: threading.Event,
    run_id: str,
) -> None:
    end_message = "Stopped"
    slot = LatestFrameSlot()
    capture_done = threading.Event()
    capture_msg: dict[str, Optional[str]] = {"value": None}

    def capture_worker() -> None:
        fail_count = 0
        cap: Optional[cv2.VideoCapture] = None
        try:
            cap = open_capture(source)
            if not cap.isOpened():
                capture_msg["value"] = source_open_error(source)
                return

            is_live = source.isdigit() or source.startswith(
                ("rtsp://", "rtmp://", "http://", "https://")
            )
            frame_interval = 0.0
            if not is_live:
                try:
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                except Exception:
                    fps = 0.0
                if fps > 0.5:
                    frame_interval = 1.0 / fps

            with state.lock:
                if state.run_id != run_id:
                    return
                state.cap = cap
                state.message = "Running"

            next_time = time.monotonic()
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    fail_count += 1
                    if fail_count > 200:
                        capture_msg["value"] = "Stream ended or repeatedly failed"
                        break
                    time.sleep(0.02)
                    continue
                fail_count = 0
                slot.put(frame)

                if frame_interval > 0:
                    next_time += frame_interval
                    sleep_for = next_time - time.monotonic()
                    if sleep_for > 0:
                        if stop_event.wait(timeout=sleep_for):
                            break
                    else:
                        next_time = time.monotonic()
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            capture_done.set()

    cap_thread = threading.Thread(target=capture_worker, daemon=True)
    cap_thread.start()

    last_id = 0
    try:
        while not stop_event.is_set():
            frame, last_id = slot.get_if_new(last_id)
            if frame is None:
                if capture_done.is_set():
                    break
                time.sleep(0.005)
                continue

            annotated, stats = recognizer.process(frame)
            encoded_ok, jpeg = cv2.imencode(
                ".jpg",
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), 82],
            )
            if not encoded_ok:
                continue

            with state.lock:
                if state.run_id != run_id:
                    break
                state.latest_jpeg = jpeg.tobytes()
                state.latest_stats = asdict(stats)
                state.message = "Running"
                state.running = True
    finally:
        stop_event.set()
        cap_thread.join(timeout=5.0)
        if capture_msg["value"]:
            end_message = capture_msg["value"]
        with state.lock:
            if state.run_id == run_id:
                state.cap = None
                state.running = False
                state.message = end_message


def known_workers_payload() -> list[dict]:
    encodings_path = BASE_DIR / "encodings.pkl"
    if not encodings_path.exists():
        return []
    known = load_known_faces(str(encodings_path))
    counts: dict[str, int] = {}
    for worker_id in known.worker_ids:
        counts[worker_id] = counts.get(worker_id, 0) + 1
    rows = []
    for worker_id, record in known.workers.items():
        rows.append(
            {
                "worker_id": worker_id,
                "name": record.name,
                "role": record.role,
                "zone": record.zone,
                "shift": record.shift,
                "encodings": counts.get(worker_id, 0),
            }
        )
    return rows


@app.get("/api/health")
def health() -> dict:
    required = {
        "config": CONFIG_PATH.exists(),
        "encodings": (BASE_DIR / "encodings.pkl").exists(),
        "face_model": (BASE_DIR / "models/yolov8n-face.pt").exists(),
        "person_model": (BASE_DIR / "models/yolov8n.pt").exists(),
        "sface_model": (
            BASE_DIR / "models/face_recognition_sface_2021dec.onnx"
        ).exists(),
    }
    with state.lock:
        running = state.running
        message = state.message
    return {
        "ok": all(required.values()),
        "running": running,
        "message": message,
        "required": required,
    }


@app.get("/api/workers")
def workers() -> dict:
    return {"workers": known_workers_payload()}


@app.get("/api/defaults")
def defaults() -> dict:
    cfg = load_config(str(CONFIG_PATH)) if CONFIG_PATH.exists() else {}
    stream_cfg = cfg.get("stream", {}) or {}
    rec_cfg = cfg.get("recognition", {}) or {}
    person_cfg = cfg.get("person_detection", {}) or {}
    fallback_video = str(stream_cfg.get("fallback_video", ""))
    return {
        "streamUrl": str(stream_cfg.get("rtsp_url", "")),
        "fallbackVideo": fallback_video,
        "fallbackVideoExists": bool(fallback_video and Path(fallback_video).exists()),
        "cameraId": str(stream_cfg.get("camera_id", "CAM-01")),
        "zone": str(stream_cfg.get("zone", "Zone-A")),
        "threshold": float(rec_cfg.get("cosine_threshold", 0.45)),
        "detectEvery": int(rec_cfg.get("detect_every_n_frames", 2)),
        "faceImgSize": int(rec_cfg.get("yolo_imgsz", 1280)),
        "faceConf": float(rec_cfg.get("yolo_conf", 0.25)),
        "minFace": int(rec_cfg.get("min_face_for_embed", 32)),
        "emptyZoneFrames": int(rec_cfg.get("empty_zone_threshold_frames", 3)),
        "personEnabled": bool(person_cfg.get("enabled", True)),
        "personConf": float(person_cfg.get("conf", 0.5)),
    }


@app.post("/api/videos/upload")
def upload_video(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename received")
    original_name = Path(file.filename).name
    safe_name = "".join(
        c if c.isalnum() or c in {".", "-", "_"} else "_"
        for c in original_name.replace(" ", "_")
    )
    target = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out, length=8 * 1024 * 1024)
    return {
        "source": str(target),
        "filename": target.name,
        "size_bytes": target.stat().st_size,
    }


@app.post("/api/recognition/start")
def start_recognition(req: StartRequest) -> dict:
    source = req.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="Source is required")

    recognizer = get_recognizer()
    stop_current(wait=True)
    apply_settings(recognizer, req)
    reset_recognizer_runtime(recognizer)

    stop_event = threading.Event()
    run_id = str(time.time_ns())
    thread = threading.Thread(
        target=recognition_loop,
        args=(source, recognizer, stop_event, run_id),
        daemon=True,
    )

    with state.lock:
        state.stop_event = stop_event
        state.cap = None
        state.thread = thread
        state.run_id = run_id
        state.running = True
        state.message = "Opening stream"
        state.latest_jpeg = None
        state.latest_stats = default_stats()
        state.latest_stats["empty_zone_threshold"] = (
            recognizer.empty_zone_threshold_frames
        )

    thread.start()
    return {"running": True, "message": "Started", "source": source}


@app.post("/api/recognition/stop")
def stop_recognition() -> dict:
    stop_current(wait=True)
    with state.lock:
        state.message = "Stopped"
    return {"running": False, "message": "Stopped"}


@app.post("/api/recognition/settings")
def update_recognition_settings(req: SettingsRequest) -> dict:
    recognizer = get_recognizer()
    apply_settings(recognizer, req)
    with state.lock:
        state.latest_stats["empty_zone_threshold"] = (
            recognizer.empty_zone_threshold_frames
        )
        running = state.running
    return {
        "running": running,
        "emptyZoneFrames": recognizer.empty_zone_threshold_frames,
    }


@app.get("/api/recognition/status")
def recognition_status() -> dict:
    with state.lock:
        return {
            "running": state.running,
            "message": state.message,
            "stats": state.latest_stats,
        }


@app.get("/api/recognition/frame")
def recognition_frame() -> Response:
    with state.lock:
        jpeg = state.latest_jpeg
    if jpeg is None:
        return Response(status_code=204)
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )
