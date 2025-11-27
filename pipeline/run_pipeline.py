# pipeline/run_pipeline.py
import fire
import json
import os
from pathlib import Path
from pipeline.fetch import fetch_image
from pipeline.segment import segment_solar
from pipeline.buffer_and_area import apply_buffer_logic
from pipeline.qc import run_qc_engine
from pipeline.overlay import create_audit_overlay
from pipeline.temporal import temporal_verify

def run(
    input_folder="input",
    output_folder="submission_output",
    use_temporal=True,
    super_res=True,
):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/artifacts", exist_ok=True)

    results = []
    input_path = Path(input_folder)

    for csv_file in input_path.glob("*.csv"):
        import pandas as pd
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            lat, lon = row["latitude"], row["longitude"]

            # 1. Fetch images
            img_2025 = fetch_image(lat, lon, year=2025, super_res=super_res)
            img_2023 = fetch_image(lat, lon, year=2023, super_res=super_res) if use_temporal else None

            # 2. Segment solar panels
            masks_2025 = segment_solar(img_2025)
            masks_2023 = segment_solar(img_2023) if img_2023 else []

            # 3. Temporal proof (bonus)
            new_install = temporal_verify(masks_2023, masks_2025)

            # 4. 100% correct FAQ #4 buffer logic
            result = apply_buffer_logic(masks_2025, lat, lon)

            # 5. QC Engine → VERIFIABLE / NOT_VERIFIABLE
            result.update(run_qc_engine(img_2025, result))

            # 6. Final required fields
            final_result = {
                "sample_id": sample_id,
                "has_solar": bool(result.get("chosen_panel")),
                "pv_area_sqm_est": round(result.get("chosen_area", 0), 2),
                "verifiable": result.get("qc_status", "NOT_VERIFIABLE"),
                "buffer_radius_sqft": result.get("buffer_used", 2400),
                "temporal_proof": new_install
            }

            # 7. Save audit overlay
            overlay_path = f"{output_folder}/artifacts/{sample_id}.jpg"
            create_audit_overlay(img_2025, img_2023, result, save_path=overlay_path)

            results.append(final_result)

    # Save official predictions.json
    with open(f"{output_folder}/predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"SUCCESS! Generated {len(results)} results → {output_folder}/predictions.json")

if __name__ == "__main__":
    fire.Fire(run)
