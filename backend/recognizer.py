"""
ZEEX AI - Recognizer
Per-frame recognition orchestrator used by the FastAPI backend. Keeps a tiny
IoU tracker so we only re-detect every N frames but still update labels every
frame -> smooth UI without paying detection cost.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_pipeline import (
    FaceDetector,
    FaceEmbedder,
    KnownFaces,
    MatchResult,
    PersonDetector,
    draw_label,
    load_known_faces,
    match_embedding,
    resize_keep_aspect,
)


GREEN = (0, 200, 0)
RED = (0, 0, 220)
ORANGE = (0, 140, 240)   # Person detected but no face visible
YELLOW = (0, 200, 220)


@dataclass
class TrackedFace:
    box: Tuple[int, int, int, int]
    label: str
    color: Tuple[int, int, int]
    worker_id: Optional[str]
    score: float
    last_seen: int  # frame index


@dataclass
class TrackedPerson:
    """A person detection that has NO face matched inside it (back-of-head etc)."""
    box: Tuple[int, int, int, int]
    score: float
    last_seen: int


@dataclass
class FrameStats:
    fps: float = 0.0
    detections: int = 0
    known_count: int = 0
    unknown_count: int = 0
    person_no_face_count: int = 0
    total_people_count: int = 0
    no_person_frame_count: int = 0
    empty_zone_threshold: int = 3
    empty_zone: bool = True
    zone_status: str = "EMPTY_ZONE"


class EventLogger:
    """Writes recognition events to a text log and SQLite, with per-worker cooldown."""

    def __init__(self, log_file: Optional[str], db_file: Optional[str],
                 camera_id: str, zone: str, cooldown_seconds: float):
        self.log_file = Path(log_file) if log_file else None
        self.db_file = Path(db_file) if db_file else None
        self.camera_id = camera_id
        self.zone = zone
        self.cooldown = cooldown_seconds
        self._last_logged: Dict[str, float] = {}
        self._db: Optional[sqlite3.Connection] = None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.db_file:
            self.db_file.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.db_file), check_same_thread=False)
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    worker_id TEXT,
                    name TEXT,
                    camera_id TEXT,
                    zone TEXT,
                    score REAL
                )
                """
            )
            self._db.commit()

    def log(self, match: MatchResult) -> bool:
        """Returns True if the event was actually written (cooldown not active)."""
        key = match.worker_id or "UNKNOWN"
        now = time.time()
        last = self._last_logged.get(key, 0.0)
        if now - last < self.cooldown:
            return False
        self._last_logged[key] = now
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        wid = match.worker_id or ""
        name = match.name
        line = (
            f"{ts} | worker_id={wid or '-':<10} | name={name:<25} | "
            f"camera={self.camera_id} | zone={self.zone} | score={match.score:.3f}"
        )
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self._db is not None:
            self._db.execute(
                "INSERT INTO events(ts, worker_id, name, camera_id, zone, score)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (ts, wid, name, self.camera_id, self.zone, float(match.score)),
            )
            self._db.commit()
        return True

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None


class FrameRecognizer:
    """Wraps detector + embedder + matcher with a small IoU tracker."""

    def __init__(self, cfg: dict, base_dir: Path):
        paths = cfg["paths"]
        rec_cfg = cfg["recognition"]
        log_cfg = cfg["logging"]
        stream_cfg = cfg["stream"]
        person_cfg = cfg.get("person_detection", {}) or {}

        self.process_width = int(rec_cfg.get("process_width", 0))
        self.detect_every_n = max(1, int(rec_cfg.get("detect_every_n_frames", 1)))
        self.threshold = float(rec_cfg["cosine_threshold"])
        # Minimum face size (pixels, processed-frame coords) below which we
        # don't bother running the embedder - SFace embeddings on tiny
        # crops are noise. Far faces are still BOXED so the operator sees them.
        self.min_face_for_embed = int(rec_cfg.get("min_face_for_embed", 28))
        self.empty_zone_threshold_frames = max(
            1,
            int(rec_cfg.get("empty_zone_threshold_frames", 3)),
        )

        self.detector = FaceDetector(
            str((base_dir / paths["yolo_face_model"]).resolve()),
            conf=float(rec_cfg["yolo_conf"]),
            imgsz=int(rec_cfg.get("yolo_imgsz", 1280)),
        )
        self.embedder = FaceEmbedder(
            str((base_dir / paths["sface_model"]).resolve())
        )
        self.known: KnownFaces = load_known_faces(
            str((base_dir / paths["encodings_file"]).resolve())
        )

        # Optional person detector (catches back-of-head / occluded face)
        self.person_enabled = bool(person_cfg.get("enabled", False))
        self.person_detector: Optional[PersonDetector] = None
        self.person_model_path = str((
            base_dir / person_cfg.get("model", "models/yolov8n.pt")
        ).resolve())
        self.person_conf = float(person_cfg.get("conf", 0.5))
        self.person_imgsz = int(person_cfg.get("imgsz", 960))
        self.person_min_height_px = int(person_cfg.get("min_height_px", 70))
        self.person_min_aspect_ratio = float(person_cfg.get("min_aspect_ratio", 1.4))
        self.person_max_aspect_ratio = float(person_cfg.get("max_aspect_ratio", 4.5))
        self.person_max_area_frac = float(person_cfg.get("max_area_frac", 0.55))

        self.logger: Optional[EventLogger] = None
        if log_cfg.get("enable_file_log") or log_cfg.get("enable_sqlite"):
            self.logger = EventLogger(
                log_file=str((base_dir / paths["events_log"]).resolve())
                    if log_cfg.get("enable_file_log") else None,
                db_file=str((base_dir / paths["events_db"]).resolve())
                    if log_cfg.get("enable_sqlite") else None,
                camera_id=str(stream_cfg.get("camera_id", "CAM")),
                zone=str(stream_cfg.get("zone", "")),
                cooldown_seconds=float(log_cfg.get("cooldown_seconds", 30)),
            )

        # State
        self.frame_idx = 0
        self.tracked: List[TrackedFace] = []
        self.tracked_persons: List[TrackedPerson] = []
        self._no_person_frame_count = 0
        self._t_prev = time.time()
        self._fps = 0.0

    @property
    def n_known_workers(self) -> int:
        return len(self.known.workers)

    def update_camera_zone(self, camera_id: Optional[str] = None,
                           zone: Optional[str] = None) -> None:
        """Allow the UI to change the labels going into events.log on the fly."""
        if self.logger is None:
            return
        if camera_id is not None:
            self.logger.camera_id = camera_id
        if zone is not None:
            self.logger.zone = zone

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()

    def _ensure_person_detector(self) -> PersonDetector:
        if self.person_detector is None:
            self.person_detector = PersonDetector(
                self.person_model_path,
                conf=self.person_conf,
                imgsz=self.person_imgsz,
                min_height_px=self.person_min_height_px,
                min_aspect_ratio=self.person_min_aspect_ratio,
                max_aspect_ratio=self.person_max_aspect_ratio,
                max_area_frac=self.person_max_area_frac,
            )
        return self.person_detector

    def _label_for(self, m: MatchResult) -> Tuple[str, Tuple[int, int, int]]:
        if m.worker_id is None:
            return "UNKNOWN", RED
        rec = m.record
        zone_part = f" | {rec.zone}" if rec and rec.zone else ""
        return f"[{m.worker_id}] {m.name}{zone_part}  ({m.score:.2f})", GREEN

    @staticmethod
    def _face_inside_person(face_box: Tuple[int, int, int, int],
                            person_box: Tuple[int, int, int, int]) -> bool:
        """Is the face's CENTER inside the person box, AND does the face
        sit in roughly the upper half of the person body?"""
        fx1, fy1, fx2, fy2 = face_box
        px1, py1, px2, py2 = person_box
        cx = (fx1 + fx2) / 2.0
        cy = (fy1 + fy2) / 2.0
        if not (px1 <= cx <= px2 and py1 <= cy <= py2):
            return False
        # face should be above the person box's vertical midpoint
        return cy <= (py1 + (py2 - py1) * 0.65)

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, FrameStats]:
        """Process a single BGR frame. Returns (annotated_frame, stats)."""
        self.frame_idx += 1
        proc, scale = resize_keep_aspect(frame_bgr, self.process_width)

        is_detect_frame = (self.frame_idx == 1
                           or (self.frame_idx % self.detect_every_n == 0))

        if is_detect_frame:
            # ---- Face detection + recognition ----
            face_boxes = self.detector.detect(proc)
            new_tracked: List[TrackedFace] = []
            for (x1, y1, x2, y2, _conf) in face_boxes:
                bw = x2 - x1; bh = y2 - y1
                # Tiny faces: still draw a box so the operator sees them, but
                # don't try to ID them - SFace on a 15-px crop is just noise.
                if min(bw, bh) < self.min_face_for_embed:
                    new_tracked.append(TrackedFace(
                        box=(x1, y1, x2, y2),
                        label="FACE (too far)",
                        color=YELLOW,
                        worker_id=None,
                        score=0.0,
                        last_seen=self.frame_idx,
                    ))
                    continue
                emb = self.embedder.embed(proc, (x1, y1, x2, y2))
                if emb is None:
                    continue
                m = match_embedding(emb, self.known, self.threshold)
                label, color = self._label_for(m)
                new_tracked.append(TrackedFace(
                    box=(x1, y1, x2, y2),
                    label=label,
                    color=color,
                    worker_id=m.worker_id,
                    score=m.score,
                    last_seen=self.frame_idx,
                ))
                if self.logger is not None:
                    self.logger.log(m)
            self.tracked = new_tracked

            # ---- Person detection (back-of-head fallback) ----
            new_persons: List[TrackedPerson] = []
            if self.person_enabled:
                person_boxes = self._ensure_person_detector().detect(proc)
                for (px1, py1, px2, py2, pconf) in person_boxes:
                    has_face = any(
                        self._face_inside_person(t.box, (px1, py1, px2, py2))
                        for t in self.tracked
                    )
                    if has_face:
                        continue  # face label already covers this person
                    new_persons.append(TrackedPerson(
                        box=(px1, py1, px2, py2),
                        score=pconf,
                        last_seen=self.frame_idx,
                    ))
            self.tracked_persons = new_persons
        # else: keep previous self.tracked / self.tracked_persons

        # FPS (EMA)
        now = time.time()
        dt = now - self._t_prev
        self._t_prev = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0 else (0.8 * self._fps + 0.2 * inst)

        # Draw on the original frame; upscale boxes from proc-coords if needed.
        out = frame_bgr.copy()
        inv = 1.0 / scale if scale != 0 else 1.0

        # 1) draw "person without face" boxes FIRST (so face labels go on top
        #    if there's any overlap/border)
        for tp in self.tracked_persons:
            x1, y1, x2, y2 = tp.box
            if scale != 1.0:
                x1 = int(x1 * inv); y1 = int(y1 * inv)
                x2 = int(x2 * inv); y2 = int(y2 * inv)
            draw_label(out, (x1, y1, x2, y2),
                       f"PERSON (no face)  ({tp.score:.2f})", ORANGE)

        # 2) draw face boxes
        known_n = unknown_n = far_n = 0
        for t in self.tracked:
            x1, y1, x2, y2 = t.box
            if scale != 1.0:
                x1 = int(x1 * inv); y1 = int(y1 * inv)
                x2 = int(x2 * inv); y2 = int(y2 * inv)
            draw_label(out, (x1, y1, x2, y2), t.label, t.color)
            if t.color == YELLOW:
                far_n += 1
            elif t.worker_id is None:
                unknown_n += 1
            else:
                known_n += 1

        person_no_face_n = len(self.tracked_persons)
        total_people_n = known_n + unknown_n + far_n + person_no_face_n
        if total_people_n == 0:
            self._no_person_frame_count += 1
        else:
            self._no_person_frame_count = 0
        empty_zone = self._no_person_frame_count >= self.empty_zone_threshold_frames
        if empty_zone:
            zone_status = "EMPTY_ZONE"
        elif self._no_person_frame_count > 0:
            zone_status = "CHECKING_EMPTY_ZONE"
        else:
            zone_status = "OCCUPIED_ZONE"

        # HUD
        hud = (f"status:{zone_status}  FPS:{self._fps:5.1f}  faces:{len(self.tracked)} "
               f"(known:{known_n} unknown:{unknown_n} far:{far_n})  "
               f"persons-no-face:{person_no_face_n}  "
               f"empty:{self._no_person_frame_count}/{self.empty_zone_threshold_frames}  "
               f"thr:{self.threshold:.2f}")
        cv2.rectangle(out, (0, 0), (out.shape[1], 24), (32, 32, 32), -1)
        cv2.putText(out, hud, (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if empty_zone:
            text = "EMPTY ZONE / NO PERSON"
            (tw, th), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            pad_x = 18
            pad_y = 12
            x1 = max(0, (out.shape[1] - tw) // 2 - pad_x)
            y1 = max(30, (out.shape[0] - th) // 2 - pad_y)
            x2 = min(out.shape[1] - 1, x1 + tw + 2 * pad_x)
            y2 = min(out.shape[0] - 1, y1 + th + 2 * pad_y)
            cv2.rectangle(out, (x1, y1), (x2, y2), (22, 96, 130), -1)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 190, 255), 2)
            cv2.putText(out, text, (x1 + pad_x, y2 - pad_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        return out, FrameStats(
            fps=self._fps,
            detections=len(self.tracked),
            known_count=known_n,
            unknown_count=unknown_n,
            person_no_face_count=person_no_face_n,
            total_people_count=total_people_n,
            no_person_frame_count=self._no_person_frame_count,
            empty_zone_threshold=self.empty_zone_threshold_frames,
            empty_zone=empty_zone,
            zone_status=zone_status,
        )
