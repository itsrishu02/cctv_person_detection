"""
ZEEX AI - Face Pipeline
Shared building blocks: YOLOv8n-face detector + SFace embedder + cosine matcher.
Used by encode_faces.py and the FastAPI recognition backend.
"""
from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


# ---------- Config ----------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- Detection ----------

class FaceDetector:
    """YOLOv8n-face wrapper. Returns (x1, y1, x2, y2, conf) boxes.

    `imgsz` is the inference resolution. Default YOLO is 640, which loses
    tiny/distant faces. 1280-1536 is the sweet spot for factory CCTV where
    workers are often 30-50 px tall.
    """

    def __init__(self, model_path: str, conf: float = 0.4,
                 imgsz: int = 1280, iou: float = 0.5, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        results = self.model.predict(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        boxes_out: List[Tuple[int, int, int, int, float]] = []
        if not results:
            return boxes_out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return boxes_out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        h, w = frame_bgr.shape[:2]
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
            if x2 > x1 and y2 > y1:
                boxes_out.append((x1, y1, x2, y2, float(c)))
        return boxes_out


class PersonDetector:
    """Generic YOLOv8n (COCO) wrapper, filtered to the 'person' class.

    Used to flag people whose face we cannot see (back of head, extreme
    profile, occluded, too small for the face model). When a person box
    has no face matched inside it, we still draw it as 'PERSON (no face)'.
    """

    PERSON_CLASS_ID = 0  # COCO

    def __init__(self, model_path: str, conf: float = 0.5,
                 imgsz: int = 960, iou: float = 0.5,
                 min_height_px: int = 70,
                 min_aspect_ratio: float = 1.4,
                 max_aspect_ratio: float = 4.5,
                 max_area_frac: float = 0.55,
                 device: str = "cpu"):
        from ultralytics import YOLO

        """
        Filters applied AFTER YOLO to suppress object-as-person false positives:
          - conf            : raise to 0.5 by default (was 0.4) - cuts most FPs.
          - min_height_px   : drop boxes shorter than this in processed pixels.
                              Real people are rarely <70 px even at 50 m on 1080p.
          - min/max_aspect_ratio: real persons are tall+narrow (h/w ~ 1.4-4.0).
                              Yarn-cone racks come back wide (h/w < 1).
                              Pillars come back extremely tall (h/w > 5).
          - max_area_frac   : drop boxes covering >55% of the frame
                              (usually a foreground machine/wall misfire).
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou
        self.min_height_px = min_height_px
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_area_frac = max_area_frac
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        results = self.model.predict(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
            device=self.device,
        )
        out: List[Tuple[int, int, int, int, float]] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            # Shape-based geometric filters
            if bh < self.min_height_px:
                continue
            ar = bh / float(bw)  # height / width
            if ar < self.min_aspect_ratio or ar > self.max_aspect_ratio:
                continue
            if frame_area > 0 and (bw * bh) / frame_area > self.max_area_frac:
                continue
            out.append((x1, y1, x2, y2, float(c)))
        return out


# ---------- Embedding (SFace) ----------

class FaceEmbedder:
    """OpenCV SFace recognizer. Outputs 128-d embeddings.

    SFace expects an aligned 112x112 face crop. We approximate alignment using
    the YOLO bounding box (no landmarks). For the prototype this works well
    when faces are roughly upright and centered, which is the case for
    workers facing the camera in a factory zone.
    """

    EMBED_DIM = 128

    def __init__(self, model_path: str):
        # backend / target = 0 -> default (CPU). FaceRecognizerSF takes
        # (model, config, backend_id, target_id).
        self.recognizer = cv2.FaceRecognizerSF.create(model_path, "", 0, 0)

    def _preprocess_crop(self, frame_bgr: np.ndarray,
                         box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = box
        # Expand box by 15% to include forehead/chin for better embedding stability.
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return None
        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.15)
        h, w = frame_bgr.shape[:2]
        cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x); cy2 = min(h, y2 + pad_y)
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)

    def embed(self, frame_bgr: np.ndarray,
              box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        aligned = self._preprocess_crop(frame_bgr, box)
        if aligned is None:
            return None
        feat = self.recognizer.feature(aligned)  # (1, 128) float32
        # Normalise so cosine sim = dot product
        v = feat.flatten().astype(np.float32)
        n = np.linalg.norm(v)
        if n < 1e-9:
            return None
        return v / n


# ---------- Matching ----------

@dataclass
class WorkerRecord:
    worker_id: str
    name: str
    role: str = ""
    zone: str = ""
    shift: str = ""


@dataclass
class KnownFaces:
    """Holds the encoded gallery: stacked embeddings + parallel metadata list."""
    embeddings: np.ndarray            # (N, 128) L2-normalised
    worker_ids: List[str]             # length N (one per embedding)
    workers: dict                     # worker_id -> WorkerRecord

    @property
    def is_empty(self) -> bool:
        return self.embeddings.size == 0


def save_known_faces(known: KnownFaces, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(
            {
                "embeddings": known.embeddings,
                "worker_ids": known.worker_ids,
                "workers": {wid: vars(rec) for wid, rec in known.workers.items()},
            },
            f,
        )


def load_known_faces(path: str) -> KnownFaces:
    with open(path, "rb") as f:
        data = pickle.load(f)
    workers = {wid: WorkerRecord(**rec) for wid, rec in data["workers"].items()}
    return KnownFaces(
        embeddings=np.asarray(data["embeddings"], dtype=np.float32),
        worker_ids=list(data["worker_ids"]),
        workers=workers,
    )


@dataclass
class MatchResult:
    worker_id: Optional[str]   # None if unknown
    name: str                  # "UNKNOWN" if no match
    score: float               # best cosine similarity in [-1, 1]
    record: Optional[WorkerRecord]


def match_embedding(query: np.ndarray, known: KnownFaces,
                    threshold: float) -> MatchResult:
    """Cosine similarity match. query and known.embeddings must be L2-normalised."""
    if known.is_empty:
        return MatchResult(None, "UNKNOWN", 0.0, None)
    sims = known.embeddings @ query  # (N,)
    # Aggregate by worker: take the max similarity per worker_id, then pick top.
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score < threshold:
        return MatchResult(None, "UNKNOWN", best_score, None)
    wid = known.worker_ids[best_idx]
    rec = known.workers.get(wid)
    name = rec.name if rec else wid
    return MatchResult(wid, name, best_score, rec)


# ---------- Drawing ----------

def draw_label(frame: np.ndarray, box: Tuple[int, int, int, int],
               text: str, color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Label background
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pad = 4
    label_y2 = y1
    label_y1 = max(0, y1 - th - 2 * pad)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 2 * pad, label_y2), color, -1)
    cv2.putText(
        frame, text,
        (x1 + pad, label_y2 - pad),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )


# ---------- Frame helpers ----------

def resize_keep_aspect(frame: np.ndarray, target_w: int) -> Tuple[np.ndarray, float]:
    """Resize so width == target_w. Returns (resized, scale_factor_applied)."""
    if target_w <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    if w == target_w:
        return frame, 1.0
    scale = target_w / float(w)
    new_size = (target_w, int(round(h * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA), scale
