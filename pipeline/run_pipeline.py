# ============================================================
# Rooftop Solar Panel Detection & Spatial Quantification
# EcoInnovators Ideathon 2026 – FINAL PIPELINE
# ============================================================

import os
import cv2
import math
import json
import requests
import numpy as np
import pandas as pd
from ultralytics import YOLO
from shapely.geometry import Point, box
from shapely.ops import unary_union
from PIL import Image
from io import BytesIO

# ===================== API KEYS =====================
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "") # google api
ESRI_API_KEY   = os.getenv("ESRI_API_KEY", "") #esri  api
BING_API_KEY   = os.getenv("BING_MAPS_API_KEY", "")#bing api

# ===================== PATH CONFIG =====================
MODEL_PATH = "models/best.pt"
INPUT_XLSX = "book3.xlsx" # excel 

OUTPUT_PRED_DIR = "outputs/predictions"
OUTPUT_ART_DIR  = "outputs/artifacts"

os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)
os.makedirs(OUTPUT_ART_DIR, exist_ok=True)

# ===================== CONSTANTS =====================
EARTH_CIRCUMFERENCE = 40075016.686
TILE_SIZE = 256
SQFT_TO_SQM = 0.092903

ZOOM = 19
IMAGE_SIZE = 640

# ===================== GEO HELPERS =====================
def meters_per_pixel(latitude, zoom):
    return (
        EARTH_CIRCUMFERENCE * math.cos(math.radians(latitude))
    ) / (2 ** zoom * TILE_SIZE)

def buffer_radius_pixels(buffer_sqft, latitude, zoom):
    buffer_sqm = buffer_sqft * SQFT_TO_SQM
    buffer_m = math.sqrt(buffer_sqm / math.pi)
    return buffer_m / meters_per_pixel(latitude, zoom)

# ===================== IMAGE FETCHERS =====================
def fetch_google(lat, lon):
    if not GOOGLE_API_KEY:
        raise RuntimeError("Google key missing")

    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM,
        "size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "maptype": "satellite",
        "scale": 2,
        "key": GOOGLE_API_KEY
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError("Google request failed")

    img = Image.open(BytesIO(r.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), "Google Maps"


def fetch_esri(lat, lon):
    if not ESRI_API_KEY:
        raise RuntimeError("ESRI key missing")

    url = (
        "https://services.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/export"
    )
    params = {
        "center": f"{lon},{lat}",
        "zoom": ZOOM,
        "size": f"{IMAGE_SIZE},{IMAGE_SIZE}",
        "format": "png",
        "f": "image",
        "token": ESRI_API_KEY
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError("ESRI request failed")

    img_np = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("ESRI decode failed")

    return img, "ESRI World Imagery"


def fetch_bing(lat, lon):
    if not BING_API_KEY:
        raise RuntimeError("Bing key missing")

    url = "https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial"
    params = {
        "mapSize": f"{IMAGE_SIZE},{IMAGE_SIZE}",
        "zoomLevel": ZOOM,
        "key": BING_API_KEY
    }
    r = requests.get(f"{url}/{lat},{lon}", params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError("Bing request failed")

    img = Image.open(BytesIO(r.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), "Bing Maps"


def fetch_satellite_image(lat, lon):
    for fn in [fetch_google, fetch_esri, fetch_bing]:
        try:
            return fn(lat, lon)
        except Exception:
            continue
    raise RuntimeError("All imagery sources failed")

# ===================== LOAD MODEL =====================
model = YOLO(MODEL_PATH)

# ===================== CORE PIPELINE =====================
def run_pipeline(sample_id, lat, lon):

    # ---------- IMAGE FETCH ----------
    try:
        image, source = fetch_satellite_image(lat, lon)
    except Exception as e:
        print(f"[{sample_id}] Image fetch failed:", e)
        return {
            "sample_id": sample_id,
            "latitude": lat,
            "longitude": lon,
            "has_solar": False,
            "pv_area_sqm_est": 0.0,
            "buffer_radius_sqft": None,
            "confidence": 0.0,
            "qc_status": "NOT_VERIFIABLE",
            "image_source": None,
            "artefact_path": None
        }

    raw_path = f"{OUTPUT_ART_DIR}/{sample_id}_raw.png"
    cv2.imwrite(raw_path, image)

    h, w, _ = image.shape
    cx, cy = w // 2, h // 2

    # ---------- YOLO DETECTION ----------
    results = model(image)[0]
    detections = []

    for b in results.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": float(b.conf[0])
        })

    # ---------- BUFFER + TOTAL OVERLAP LOGIC ----------
    total_overlap_px = 0.0
    selected_buffer_sqft = 2400
    buffer_px_used = buffer_radius_pixels(2400, lat, ZOOM)

    for sqft in [1200, 2400]:
        radius_px = buffer_radius_pixels(sqft, lat, ZOOM)
        buffer_circle = Point(cx, cy).buffer(radius_px)

        overlaps = []
        buffer_conf = []

        for d in detections:
            poly = box(*d["bbox"])
            inter = poly.intersection(buffer_circle)
            if not inter.is_empty:
                overlaps.append(inter)
                buffer_conf.append(d["confidence"])

        if overlaps:
            total_overlap_px = unary_union(overlaps).area
            selected_buffer_sqft = sqft
            buffer_px_used = radius_px
            confidence = float(np.mean(buffer_conf))
            break
    else:
        confidence = 0.0

    # ---------- AREA + QC ----------
    mpp = meters_per_pixel(lat, ZOOM)
    pv_area = total_overlap_px * (mpp ** 2)
    has_solar = total_overlap_px > 0

    qc_status = (
        "VERIFIABLE"
        if has_solar and confidence >= 0.5
        else "NOT_VERIFIABLE"
    )

    # ---------- ARTIFACT ----------
    overlay = image.copy()
    for d in detections:
        cv2.rectangle(overlay, d["bbox"][:2], d["bbox"][2:], (0, 255, 0), 2)

    r1200 = buffer_radius_pixels(1200, lat, ZOOM)
    r2400 = buffer_radius_pixels(2400, lat, ZOOM)

    cv2.circle(overlay, (cx, cy), int(r1200), (0, 255, 0), 4)     # GREEN
    cv2.circle(overlay, (cx, cy), int(r2400), (255, 0, 255), 4) # VIOLET

    overlay_path = f"{OUTPUT_ART_DIR}/{sample_id}_overlay.png"
    cv2.imwrite(overlay_path, overlay)

    # ---------- OUTPUT ----------
    return {
        "sample_id": sample_id,
        "latitude": lat,
        "longitude": lon,
        "has_solar": has_solar,
        "pv_area_sqm_est": round(pv_area, 2),
        "buffer_radius_sqft": selected_buffer_sqft,
        "confidence": round(confidence, 2),
        "qc_status": qc_status,
        "image_source": source,
        "artefact_path": overlay_path
    }

# ===================== ENTRY POINT =====================
if __name__ == "__main__":

    df = pd.read_excel(INPUT_XLSX)
    results = []

    for _, row in df.iterrows():
        results.append(
            run_pipeline(
                row["sample_id"],
                row["Latitude"],
                row["Longitude"]
            )
        )

    with open(f"{OUTPUT_PRED_DIR}/submission.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ PIPELINE COMPLETE")
    print("Total samples:", len(results))
    print("Verifiable cases:", sum(r["qc_status"] == "VERIFIABLE" for r in results))
