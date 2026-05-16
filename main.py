"""
CarScan AI — Unified FastAPI Backend
Serves both Page 1 (Damage Detection) and Page 2 (Cost Estimation).
"""
import io
import sys
import os
import json
import re
import base64
import uuid
import asyncio
import datetime
from pathlib import Path
from typing import List, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import uvicorn
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
STATIC_DIR = BASE_DIR / "static"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

CLASSES = [
    "damaged door", "damaged window", "damaged headlight", "damaged mirror",
    "dent", "damaged hood", "damaged bumper", "damaged wind shield",
]

CLASS_COLORS = {
    "damaged door":         (0,   0,   255),
    "damaged window":       (255, 0,   0  ),
    "damaged headlight":    (0,   165, 255),
    "damaged mirror":       (0,   255, 255),
    "dent":                 (0,   255, 0  ),
    "damaged hood":         (255, 0,   255),
    "damaged bumper":       (255, 255, 0  ),
    "damaged wind shield":  (128, 0,   128),
}

try:
    from ultralytics import YOLO as _YOLO
    det_model = _YOLO(str(MODEL_PATH))
    print(f"[OK] Ultralytics YOLOv8 model loaded from: {MODEL_PATH}")
except Exception as e:
    det_model = None
    print(f"[WARN] Model load failed: {e}")


def run_detection(cv_img: np.ndarray, conf_thresh: float = 0.25) -> dict:
    rgb = cv_img[:, :, ::-1].copy()
    results = det_model.predict(rgb, conf=conf_thresh, verbose=False)[0]
    boxes_out, confs_out, cls_out = [], [], []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x, y = int(x1), int(y1)
        w, h  = int(x2 - x1), int(y2 - y1)
        boxes_out.append([x, y, w, h])
        confs_out.append(round(conf * 100, 2))
        cls_out.append(CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}")
    return {"boxes": boxes_out, "confidences": confs_out, "classes": cls_out}


def calculate_severity(confidence: float, box: list, img_shape: tuple) -> float:
    ih, iw = img_shape
    x, y, w, h = box
    area_factor = min(1.0, (w * h) / (iw * ih) * 10)
    severity = (confidence / 100 * 0.65 + area_factor * 0.35) * 100
    return round(min(100.0, severity), 1)


def annotate_image(image: np.ndarray, results: dict) -> np.ndarray:
    annotated = image.copy()
    overlay   = annotated.copy()
    font      = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft, bt = 0.55, 1, 2
    for box, conf, cls in zip(results["boxes"], results["confidences"], results["classes"]):
        x, y, w, h = box
        color  = CLASS_COLORS.get(cls, (0, 255, 0))
        label  = f"{cls}  {conf:.1f}%"
        (tw, th), bl = cv2.getTextSize(label, font, fs, ft)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, bt, lineType=cv2.LINE_AA)
        lx1, ly1 = x, max(y - th - bl - 4, 0)
        lx2, ly2 = x + tw + 6, y
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        overlay = annotated.copy()
        cv2.putText(annotated, label, (x+3, y - bl - 2),
                    font, fs, (255, 255, 255), ft, lineType=cv2.LINE_AA)
    return annotated


app = FastAPI(title="CarScan AI", description="Car Damage Detection & Cost Estimation")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

app.mount("/static",  StaticFiles(directory=str(STATIC_DIR)),  name="static")
app.mount("/assets",  StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


@app.get("/", response_class=HTMLResponse)
async def page_index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/estimate", response_class=HTMLResponse)
async def page_estimate():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/detect")
async def detect(
    image:        UploadFile = File(...),
    brand:        str = Form(...),
    model_name:   str = Form(...),
    year:         str = Form(...),
    trim:         str = Form(default=""),
    color:        str = Form(default=""),
    owner_name:   str = Form(default=""),
    registration: str = Form(default=""),
    city:         str = Form(default="India"),
    claim_id:     str = Form(default=""),
):
    if det_model is None:
        raise HTTPException(500, "Detection model not loaded. Check model path.")
    img_bytes = await image.read()
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    cv_img    = np.array(pil_img)[:, :, ::-1].copy()
    results    = run_detection(cv_img)
    detections = []
    for box, conf, cls in zip(results["boxes"], results["confidences"], results["classes"]):
        detections.append({
            "class":      cls,
            "confidence": round(conf, 2),
            "severity":   calculate_severity(conf, box, cv_img.shape[:2]),
            "box":        box,
        })
    ann_img = annotate_image(cv_img, results)
    _, ann_buf = cv2.imencode(".png", ann_img)
    _, ori_buf = cv2.imencode(".png", cv_img)
    if not claim_id:
        claim_id = f"CLM-{datetime.date.today().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
    return {
        "claim_id":        claim_id,
        "annotated_image": base64.b64encode(ann_buf.tobytes()).decode(),
        "original_image":  base64.b64encode(ori_buf.tobytes()).decode(),
        "detections":      detections,
        "vehicle": {
            "brand": brand, "model_name": model_name, "year": year,
            "trim": trim, "color": color, "owner_name": owner_name,
            "registration": registration, "city": city,
        },
        "total_damages": len(detections),
        "timestamp":     datetime.datetime.now().isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
