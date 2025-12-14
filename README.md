🏠 Rooftop Solar Panel Detection & Spatial Quantification
EcoInnovators Ideathon 2026 – College Track

This repository implements a governance-grade, audit-friendly pipeline to verify rooftop solar PV installations using satellite imagery, computer vision, and geospatial reasoning.

The solution is designed to prioritize correctness, auditability, and ethical decision-making over aggressive detection.

📌 Problem Statement

Given a list of locations (latitude, longitude), the system must:

Retrieve recent high-resolution satellite imagery

Detect rooftop solar PV using a trained YOLOv8 model

Apply progressive buffer logic (1200 → 2400 sq.ft)

Estimate total PV area (m²) within the smallest qualifying buffer

Output VERIFIABLE / NOT_VERIFIABLE quality control status

Generate audit-friendly artifacts (images + JSON)

🧠 Key Design Principles

Detection is probabilistic → verification is conservative

Buffers are defined in real-world units (meters), not pixels

Area is computed using meters-per-pixel scaling

NOT_VERIFIABLE is a valid and expected output

Avoids false positives in governance contexts

🗂 Repository Structure
Eva-Innovators-Evara-Innovations/
│
├── models/
│   └── best.pt                 # Trained YOLOv8 solar panel model
│
├── outputs/
│   ├── predictions/            # JSON outputs
│   └── artifacts/              # Raw & overlay images
│
├── pipeline.py                 # Main end-to-end pipeline
├── book3.xlsx                  # Input file (sample_id, lat, lon)
├── requirements.txt
└── README.md

📥 Input Format (.xlsx)

The input Excel file must contain:

Column Name	Description
sample_id	Unique identifier
Latitude	Latitude (WGS84)
Longitude	Longitude (WGS84)
🌍 Supported Satellite Image Sources

The pipeline automatically falls back between APIs.

Provider	API Key Required	Notes
Google Maps Static API	✅	Best clarity (preferred)
ESRI World Imagery	✅	Allowed & common
Bing Maps	✅	Robust fallback

At least one valid API key must be provided.

API keys are read from environment variables:

export GOOGLE_MAPS_API_KEY="your_key"
export ESRI_API_KEY="your_key"
export BING_MAPS_API_KEY="your_key"

⚙️ Installation
pip install -r requirements.txt


Required packages include:

ultralytics

opencv-python

shapely

numpy

pandas

requests

pillow

▶️ How to Run
python pipeline.py


The pipeline will:

Read locations from Excel

Fetch satellite imagery

Run YOLOv8 inference

Apply buffer logic

Estimate PV area

Save results to outputs/

📐 Buffer Logic (EcoInnovators FAQ-Aligned)

Start with 1200 sq.ft buffer

If any PV overlap exists, stop

Else, expand to 2400 sq.ft

Compute TOTAL overlapping PV area

Convert pixel area → m² using:

meters_per_pixel = (EarthCircumference × cos(latitude)) / (2^zoom × tile_size)


This ensures scale correctness regardless of zoom or image size.

📊 Output JSON (Per Sample)
{
  "sample_id": "S001",
  "latitude": 12.9716,
  "longitude": 77.5946,
  "has_solar": true,
  "pv_area_sqm_est": 18.42,
  "buffer_radius_sqft": 1200,
  "confidence": 0.73,
  "qc_status": "VERIFIABLE",
  "image_source": "Google Maps",
  "artefact_path": "outputs/artifacts/S001_overlay.png"
}

🖼 Audit Artifacts

For every sample, the pipeline generates:

Raw satellite image

Overlay image with:

YOLO bounding boxes

1200 sq.ft buffer (green)

2400 sq.ft buffer (violet)

These artifacts support human audit and explainability.

✅ Quality Control (QC) Logic
QC Status	Meaning
VERIFIABLE	Clear visual evidence with sufficient confidence
NOT_VERIFIABLE	Insufficient evidence due to resolution, shadow, occlusion, or ambiguity

The system never forces a decision when evidence is weak.

🧪 Known Limitations (Explicitly Handled)

Small or shadowed panels may not be detectable

Low-resolution imagery may result in NOT_VERIFIABLE

Detection misses do not invalidate buffer logic

These are documented behaviors, not bugs.

🏆 Evaluation Alignment
Criterion	How Addressed
Detection Accuracy (40%)	YOLOv8 + conservative thresholds
Quantification RMSE (20%)	Total overlap-based area
Robustness (20%)	Multi-API fallback + QC
Code & Ethics (20%)	Audit artifacts + NOT_VERIFIABLE handling

“Our system combines probabilistic computer vision with meter-accurate geospatial buffers and explicit uncertainty handling to deliver audit-ready rooftop solar verification.”
