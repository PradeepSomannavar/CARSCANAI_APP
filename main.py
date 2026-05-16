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

import pdfplumber
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable
)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from storage import upload_to_cloudinary, insert_car_scan

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


DARK   = HexColor("#0D1117")
CARD   = HexColor("#161B22")
ACCENT = HexColor("#6C63FF")
CYAN   = HexColor("#00D4FF")
WHITE  = HexColor("#E6EDF3")
GRAY   = HexColor("#8B949E")
BORDER = HexColor("#30363D")
RED    = HexColor("#FF4444")
ORANGE = HexColor("#FF8C00")
GREEN  = HexColor("#00C853")


def generate_report_pdf(data: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm,  bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    elems  = []

    vehicle    = data.get("vehicle", {})
    detections = data.get("detections", [])
    claim_id   = data.get("claim_id", "N/A")
    ts         = data.get("timestamp", datetime.datetime.now().isoformat())[:10]
    ann_b64    = data.get("annotated_image", "")

    hdr = Table([[
        Paragraph('<font color="#6C63FF" size="18"><b>CarScan AI</b></font>'
                  '<br/><font color="#8B949E" size="9">Automated Damage Detection</font>',
                  styles["Normal"]),
        Paragraph(f'<font color="#00D4FF" size="13"><b>DAMAGE REPORT</b></font>'
                  f'<br/><font color="#8B949E" size="8">Claim ID: {claim_id}</font>'
                  f'<br/><font color="#8B949E" size="8">Date: {ts}</font>',
                  ParagraphStyle("r", parent=styles["Normal"], alignment=TA_RIGHT))
    ]], colWidths=[9*cm, 9*cm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), DARK),
        ("BOX",          (0,0), (-1,-1), 1.5, ACCENT),
        ("TOPPADDING",   (0,0), (-1,-1), 14),
        ("BOTTOMPADDING",(0,0), (-1,-1), 14),
        ("LEFTPADDING",  (0,0), (-1,-1), 14),
        ("RIGHTPADDING", (0,0), (-1,-1), 14),
    ]))
    elems += [hdr, Spacer(1, 0.4*cm)]

    elems.append(Paragraph('<font color="#6C63FF" size="11"><b>VEHICLE INFORMATION</b></font>',
                           styles["Normal"]))
    elems.append(Spacer(1, 0.2*cm))

    v = vehicle
    vrows = [
        ["Brand",        v.get("brand","\u2014"),        "Model",        v.get("model_name","\u2014")],
        ["Year",         v.get("year","\u2014"),          "Trim",         v.get("trim","\u2014")],
        ["Color",        v.get("color","\u2014"),         "Registration", v.get("registration","\u2014")],
        ["Owner",        v.get("owner_name","\u2014"),    "City",         v.get("city","\u2014")],
    ]
    vtab = Table(vrows, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm])
    vtab.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (0,-1), CARD),
        ("BACKGROUND",   (2,0), (2,-1), CARD),
        ("BACKGROUND",   (1,0), (1,-1), DARK),
        ("BACKGROUND",   (3,0), (3,-1), DARK),
        ("TEXTCOLOR",    (0,0), (0,-1), ACCENT),
        ("TEXTCOLOR",    (2,0), (2,-1), ACCENT),
        ("TEXTCOLOR",    (1,0), (-1,-1), WHITE),
        ("FONTNAME",     (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",     (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("GRID",         (0,0), (-1,-1), 0.5, BORDER),
        ("TOPPADDING",   (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0), (-1,-1), 7),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
    ]))
    elems += [vtab, Spacer(1, 0.4*cm)]

    if ann_b64:
        elems.append(Paragraph('<font color="#6C63FF" size="11"><b>DAMAGE DETECTION IMAGE</b></font>',
                               styles["Normal"]))
        elems.append(Spacer(1, 0.2*cm))
        img_data = base64.b64decode(ann_b64)
        pil_img  = Image.open(io.BytesIO(img_data)).convert("RGB")
        max_w, max_h = 16*cm, 9*cm
        iw, ih = pil_img.size
        ratio  = min(max_w/iw, max_h/ih)
        nw, nh = int(iw*ratio), int(ih*ratio)
        pil_img = pil_img.resize((nw, nh), Image.LANCZOS)
        img_buf = io.BytesIO()
        pil_img.save(img_buf, format="JPEG", quality=92)
        img_buf.seek(0)
        rl_img = RLImage(img_buf, width=nw*72/96, height=nh*72/96)
        img_tab = Table([[rl_img]], colWidths=[18*cm])
        img_tab.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), DARK),
            ("BOX",        (0,0), (-1,-1), 1, BORDER),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("TOPPADDING", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING",(0,0),(-1,-1),10),
        ]))
        elems += [img_tab, Spacer(1, 0.4*cm)]

    elems.append(Paragraph('<font color="#6C63FF" size="11"><b>DAMAGE SUMMARY</b></font>',
                           styles["Normal"]))
    elems.append(Spacer(1, 0.2*cm))

    det_rows = [["#", "Damage Type", "Confidence", "Severity", "Status"]]
    for i, d in enumerate(detections, 1):
        sev   = d.get("severity", 0)
        status = "CRITICAL" if sev >= 70 else ("MODERATE" if sev >= 40 else "MINOR")
        det_rows.append([str(i), d.get("class","\u2014"),
                         f"{d.get('confidence',0):.1f}%", f"{sev}%", status])
    if not detections:
        det_rows.append(["\u2014", "No damage detected", "\u2014", "\u2014", "CLEAR"])

    dtab = Table(det_rows, colWidths=[1*cm, 5.5*cm, 3*cm, 3*cm, 3.5*cm])
    dstyle = [
        ("BACKGROUND", (0,0), (-1,0),  ACCENT),
        ("TEXTCOLOR",  (0,0), (-1,0),  WHITE),
        ("FONTNAME",   (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("GRID",       (0,0), (-1,-1), 0.5, BORDER),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LEFTPADDING",(0,0), (-1,-1), 8),
    ]
    for i, d in enumerate(detections, 1):
        sev   = d.get("severity", 0)
        rbg   = HexColor("#1A0A0A") if sev>=70 else (HexColor("#1A1000") if sev>=40 else HexColor("#0A1A0A"))
        scol  = RED if sev>=70 else (ORANGE if sev>=40 else GREEN)
        dstyle += [("BACKGROUND",(0,i),(-1,i),rbg),
                   ("TEXTCOLOR", (0,i),(-1,i),WHITE),
                   ("TEXTCOLOR", (4,i),(4,i), scol)]
    dtab.setStyle(TableStyle(dstyle))
    elems += [dtab, Spacer(1, 0.4*cm)]

    machine = {
        "claim_id": claim_id, "timestamp": ts,
        "vehicle": {k: v.get(k,"") for k in ("brand","model_name","year","trim","color","city")},
        "damages": [{"part": d.get("class",""), "confidence": round(d.get("confidence",0),2),
                     "severity": d.get("severity",0), "box": d.get("box",[])}
                    for d in detections],
        "total_parts_damaged": len(detections),
    }
    jstr = json.dumps(machine, indent=2)

    elems.append(Paragraph('<font color="#6C63FF" size="11"><b>MACHINE-READABLE SUMMARY</b></font>',
                           styles["Normal"]))
    elems.append(Spacer(1, 0.1*cm))
    elems.append(Paragraph('<font color="#8B949E" size="8">This JSON block is auto-parsed by the Cost Estimation pipeline on Page 2.</font>',
                           styles["Normal"]))
    elems.append(Spacer(1, 0.1*cm))

    safe_json = jstr.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    safe_json = safe_json.replace("\n","<br/>").replace(" ","&nbsp;")
    json_para = Paragraph(
        f'<font face="Courier" size="7" color="#00D4FF">{safe_json}</font>',
        ParagraphStyle("json", parent=styles["Normal"])
    )
    jbox = Table([[json_para]], colWidths=[18*cm])
    jbox.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), DARK),
        ("BOX",           (0,0),(-1,-1), 1, BORDER),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
    ]))
    elems += [jbox, Spacer(1, 0.3*cm)]

    elems.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    elems.append(Spacer(1, 0.1*cm))
    elems.append(Paragraph(
        '<font color="#8B949E" size="8">Generated by CarScan AI \u2014 Automated Vehicle Damage Detection System. '
        'This report should be reviewed by a certified automotive engineer before insurance claims are filed.</font>',
        ParagraphStyle("footer", parent=styles["Normal"], alignment=TA_CENTER)
    ))

    doc.build(elems)
    buf.seek(0)
    return buf.read()


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


@app.post("/generate-report")
@app.get("/generate-report")
async def generate_report(request: Request):
    try:
        if request.method == "GET":
            params = dict(request.query_params)
        else:
            params = await request.json()

        pdf_bytes = generate_report_pdf(params)
        cid       = params.get("claim_id", "report")

        image_url = None
        ann_b64   = params.get("annotated_image")
        if ann_b64:
            try:
                img_data = base64.b64decode(ann_b64)
                image_url = upload_to_cloudinary(img_data, "car_scans/images", f"img_{cid}")
            except Exception as img_err:
                print(f"Failed to upload image to Cloudinary: {img_err}")

        pdf_url = None
        try:
            pdf_url = upload_to_cloudinary(pdf_bytes, "car_scans/reports", f"report_{cid}")
        except Exception as pdf_err:
            print(f"Failed to upload PDF to Cloudinary: {pdf_err}")

        severity = 0.0
        detections = params.get("detections", [])
        if detections:
            severity = max([d.get("severity", 0.0) for d in detections])

        if pdf_url:
            insert_car_scan(severity, image_url, pdf_url)

        return {
            "status": "success",
            "claim_id": cid,
            "pdf_url": pdf_url,
            "image_url": image_url,
            "severity": severity
        }
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {e}")


# PDF Parsing
def extract_json_from_pdf(pdf_bytes: bytes) -> dict:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        for m in matches:
            try:
                obj = json.loads(m)
                if "damages" in obj and "vehicle" in obj:
                    return obj
            except Exception:
                continue
        return {}
    except Exception as e:
        print(f"PDF parse error: {e}")
        return {}


def merge_report_data(reports: list) -> dict:
    if not reports:
        return {}
    merged = reports[0].copy()
    all_damages = []
    for r in reports:
        all_damages.extend(r.get("damages", []))
    seen, unique = set(), []
    for d in all_damages:
        key = d.get("part", "")
        if key and key not in seen:
            seen.add(key)
            unique.append(d)
    merged["damages"] = unique
    merged["total_parts_damaged"] = len(unique)
    return merged


def build_cost_topic(merged: dict) -> str:
    v      = merged.get("vehicle", {})
    damages = merged.get("damages", [])
    brand  = v.get("brand", "Unknown")
    model  = v.get("model_name", v.get("model", "Unknown"))
    year   = v.get("year", "")
    city   = v.get("city", "India")
    dmg_list = ", ".join(d.get("part","") for d in damages) or "general damage"
    return (
        f"Car repair cost for {year} {brand} {model} with: {dmg_list}. "
        f"Find OEM and aftermarket spare part prices. Location: {city}. "
        f"Include labor rates for automotive body repair."
    )


task_store: dict = {}
executor = ThreadPoolExecutor(max_workers=4)


def _run_pipeline_thread(task_id: str, topic: str, merged: dict,
                         queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    def emit(data: dict):
        asyncio.run_coroutine_threadsafe(queue.put(data), loop)
    try:
        sys.path.insert(0, str(BASE_DIR))
        from cost_agents import (build_search_agent, build_reader_agent,
                                 cost_writer_chain, cost_critic_chain)
        vehicle = merged.get("vehicle", {})
        damages = merged.get("damages", [])
        dmg_str = ", ".join(d.get("part","") for d in damages)

        emit({"step": 1, "status": "running",
              "message": "Searching for part prices and repair costs online..."})
        search_agent = build_search_agent()
        search_res = search_agent.invoke({
            "messages": [("user",
                f"Find current repair costs and spare parts prices: {topic}. "
                "Include OEM and aftermarket prices. List specific prices in INR.")]
        })
        search_text = search_res["messages"][-1].content
        emit({"step": 1, "status": "done", "message": "Pricing data found"})

        emit({"step": 2, "status": "running",
              "message": "Scraping automotive pricing sites for detailed data..."})
        reader_agent = build_reader_agent()
        reader_res = reader_agent.invoke({
            "messages": [("user",
                f"From these search results about '{topic}', "
                f"pick the most relevant URL and scrape it for part prices.\n\n"
                f"Search Results:\n{search_text[:1000]}")]
        })
        scraped = reader_res["messages"][-1].content
        emit({"step": 2, "status": "done", "message": "Site data extracted"})

        emit({"step": 3, "status": "running",
              "message": "Generating detailed repair cost estimate..."})
        research = f"SEARCH RESULTS:\n{search_text}\n\nSCRAPED CONTENT:\n{scraped}"
        cost_report = cost_writer_chain.invoke({
            "topic":          topic,
            "research":       research,
            "vehicle_brand":  vehicle.get("brand", "Unknown"),
            "vehicle_model":  vehicle.get("model_name", vehicle.get("model", "Unknown")),
            "vehicle_year":   vehicle.get("year", ""),
            "damages":        dmg_str,
        })
        emit({"step": 3, "status": "done", "message": "Cost estimate ready"})

        emit({"step": 4, "status": "running",
              "message": "Validating estimate for accuracy and realism..."})
        vehicle_str = f"{vehicle.get('year','')} {vehicle.get('brand','')} {vehicle.get('model_name', vehicle.get('model',''))}".strip()
        feedback = cost_critic_chain.invoke({
            "report":   cost_report,
            "vehicle":  vehicle_str or "Unknown vehicle",
            "damages":  dmg_str or "general damage",
        })
        emit({"step": 4, "status": "done", "message": "Validation complete"})

        final_result = {
            "cost_report":  cost_report,
            "validation":   feedback,
            "vehicle":      vehicle,
            "damages":      damages,
            "topic":        topic,
        }
        task_store[task_id]["result"] = final_result
        emit({"step": "done", "result": final_result})
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"Pipeline error: {err}")
        emit({"step": "error", "message": str(e)})
    finally:
        task_store[task_id]["done"] = True


@app.post("/estimate-cost")
async def start_estimate(reports: List[UploadFile] = File(...)):
    task_id  = str(uuid.uuid4())
    rep_data = []
    for r in reports:
        content = await r.read()
        data    = extract_json_from_pdf(content)
        if data:
            rep_data.append(data)
    if not rep_data:
        rep_data = [{"vehicle": {}, "damages": [], "total_parts_damaged": 0}]
    merged = merge_report_data(rep_data)
    topic  = build_cost_topic(merged)
    loop  = asyncio.get_event_loop()
    queue = asyncio.Queue()
    task_store[task_id] = {"queue": queue, "loop": loop, "done": False, "result": None}
    loop.run_in_executor(
        executor,
        lambda: _run_pipeline_thread(task_id, topic, merged, queue, loop)
    )
    return {"task_id": task_id, "topic": topic, "vehicle": merged.get("vehicle", {})}


@app.get("/estimate-stream/{task_id}")
async def estimate_stream(task_id: str):
    if task_id not in task_store:
        raise HTTPException(404, "Task not found")
    async def gen() -> AsyncGenerator[str, None]:
        q = task_store[task_id]["queue"]
        while True:
            try:
                data = await asyncio.wait_for(q.get(), timeout=180.0)
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("step") in ["done", "error"]:
                    break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'step': 'keepalive'})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
