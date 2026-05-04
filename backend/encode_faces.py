"""
ZEEX AI - Encode Faces

Walks the dataset/ folder, runs YOLOv8n-face on each image to find the face,
extracts a 128-d SFace embedding, joins with workers.csv metadata, and writes
encodings.pkl for the live recognition pipeline.

Usage:
    python encode_faces.py
    python encode_faces.py --config config.yaml --out encodings.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from face_pipeline import (
    FaceDetector,
    FaceEmbedder,
    KnownFaces,
    WorkerRecord,
    load_config,
    save_known_faces,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_workers_csv(csv_path: Path) -> dict:
    """Returns {worker_id: WorkerRecord}. Missing fields default to ''."""
    if not csv_path.exists():
        print(f"[!] workers.csv not found at {csv_path} - proceeding without metadata.")
        return {}
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    if "worker_id" not in df.columns:
        raise ValueError("workers.csv must contain a 'worker_id' column")
    workers: dict = {}
    for _, row in df.iterrows():
        wid = str(row["worker_id"]).strip()
        if not wid:
            continue
        workers[wid] = WorkerRecord(
            worker_id=wid,
            name=row.get("name", "").strip(),
            role=row.get("role", "").strip(),
            zone=row.get("zone", "").strip(),
            shift=row.get("shift", "").strip(),
        )
    return workers


def pick_largest_face(boxes):
    """When an enrollment image has multiple faces, take the biggest one."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None, help="Override encodings output path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    rec_cfg = cfg["recognition"]

    base = Path(args.config).resolve().parent
    dataset_dir = (base / paths["dataset_dir"]).resolve()
    workers_csv = (base / paths["workers_csv"]).resolve()
    out_path = Path(args.out) if args.out else (base / paths["encodings_file"]).resolve()
    yolo_path = (base / paths["yolo_face_model"]).resolve()
    sface_path = (base / paths["sface_model"]).resolve()

    print(f"[i] Dataset:      {dataset_dir}")
    print(f"[i] workers.csv:  {workers_csv}")
    print(f"[i] Output:       {out_path}")
    print(f"[i] YOLO model:   {yolo_path}")
    print(f"[i] SFace model:  {sface_path}")

    if not dataset_dir.exists():
        print(f"[X] Dataset folder does not exist: {dataset_dir}")
        return 1

    workers_meta = load_workers_csv(workers_csv)
    print(f"[i] Loaded {len(workers_meta)} worker metadata rows from CSV")

    detector = FaceDetector(
        str(yolo_path),
        conf=float(rec_cfg["yolo_conf"]),
        imgsz=int(rec_cfg.get("yolo_imgsz", 1280)),
    )
    embedder = FaceEmbedder(str(sface_path))

    embeddings: list = []
    worker_ids: list = []
    enrolled_workers: dict = {}

    worker_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if not worker_dirs:
        print(f"[X] No worker subfolders found inside {dataset_dir}")
        return 1

    total_imgs = 0
    total_ok = 0
    total_no_face = 0

    for wdir in worker_dirs:
        wid = wdir.name.strip()
        rec = workers_meta.get(wid) or WorkerRecord(worker_id=wid, name=wid)
        per_worker_count = 0
        for img_path in sorted(wdir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            total_imgs += 1
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [!] Could not read {img_path.name}")
                continue
            boxes = detector.detect(img)
            best = pick_largest_face(boxes)
            if best is None:
                total_no_face += 1
                print(f"  [!] No face detected in {wid}/{img_path.name}")
                continue
            x1, y1, x2, y2, _ = best
            emb = embedder.embed(img, (x1, y1, x2, y2))
            if emb is None:
                print(f"  [!] Embedding failed on {wid}/{img_path.name}")
                continue
            embeddings.append(emb)
            worker_ids.append(wid)
            per_worker_count += 1
            total_ok += 1
        if per_worker_count > 0:
            enrolled_workers[wid] = rec
            print(f"  [+] {wid} ({rec.name or '?'}): {per_worker_count} encodings")
        else:
            print(f"  [-] {wid}: 0 encodings (skipped)")

    if not embeddings:
        print("[X] No usable face embeddings extracted. Aborting.")
        return 2

    embeddings_np = np.vstack(embeddings).astype(np.float32)
    known = KnownFaces(
        embeddings=embeddings_np,
        worker_ids=worker_ids,
        workers=enrolled_workers,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_known_faces(known, str(out_path))

    print()
    print("=" * 50)
    print(f"  Images scanned : {total_imgs}")
    print(f"  Faces encoded  : {total_ok}")
    print(f"  No-face skips  : {total_no_face}")
    print(f"  Workers enrolled: {len(enrolled_workers)}")
    print(f"  Saved -> {out_path}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
