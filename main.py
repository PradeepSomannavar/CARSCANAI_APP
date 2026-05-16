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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
