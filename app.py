from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from satquery.controller import run_analysis
from satquery.geo import read_asset, preview_rgb
from satquery.report import build_report

st.set_page_config(page_title="SatQuery AI", page_icon="🛰️", layout="wide")
st.markdown("# SatQuery AI\n### Evidence-grounded remote-sensing analysis")
st.caption("SIH26167 prototype | VQA and caption models load on first use; fallback mode remains available")

with st.sidebar:
    st.header("Analysis settings")
    st.info("Accepted: GeoTIFF, TIFF, PNG, JPEG")
    threshold = st.slider("Change threshold", 0.05, 0.60, 0.18, 0.01)
    st.caption("This threshold is recorded in the audit report.")

uploads = st.file_uploader("Upload one image or a registered pair", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=True)
query = st.text_area("Ask SatQuery AI", placeholder="What changed between these two images?", height=90)

if uploads:
    columns = st.columns(min(len(uploads), 2))
    image_records = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        for index, upload in enumerate(uploads[:2]):
            saved_path = Path(temporary_directory) / upload.name
            saved_path.write_bytes(upload.getvalue())
            try:
                asset = read_asset(str(saved_path), upload.name)
            except Exception as error:
                st.error(f"Could not read {upload.name}: {error}")
                st.stop()
            image_records.append(asset)
            with columns[index % 2]:
                st.image(preview_rgb(asset["array"]), caption=f"{upload.name} | {asset['bands']} band(s)", use_container_width=True)
                asset["modality"] = st.selectbox(
                    "Modality",
                    ["unknown", "optical", "sar"],
                    key=f"modality_{index}",
                    help="Choose SAR for radar imagery and optical for multispectral/RGB imagery.",
                )
                asset["registered"] = len(uploads) == 1 or st.checkbox(
                    "Pair is co-registered",
                    key=f"registered_{index}",
                    help="Confirm that this image is spatially aligned with the other uploaded image.",
                )

        if st.button("Analyze", type="primary", use_container_width=True):
            if not query.strip():
                st.warning("Enter a question first.")
                st.stop()
            try:
                result = run_analysis(query, image_records, {"change_threshold": threshold, "output_dir": temporary_directory})
            except Exception as error:
                st.error(str(error))
                st.stop()

            left, right = st.columns([3, 1])
            with left:
                st.subheader("Grounded answer")
                st.write(result["answer"])
            with right:
                st.metric("Confidence estimate", f"{result['confidence']:.0%}")
                st.metric("Tasks executed", len(result["plan"].tasks))

            st.subheader("Execution summary")
            st.json({"tasks": result["plan"].tasks, "intent": result["plan"].intent, "modality": result["plan"].modality, "parameters": result["plan"].parameters, "tools": [item.task for item in result["results"]], "model_metadata": [item.metadata for item in result["results"]]})
            st.subheader("Visual evidence")
            evidence_images = [item.metadata.get("evidence") for item in result["results"] if item.metadata.get("evidence", "").endswith(".png")]
            if evidence_images:
                for evidence_image in evidence_images:
                    st.image(evidence_image, caption="Change intensity map with threshold contour", use_container_width=True)
            else:
                st.info("No visual evidence was produced for this task. The system deliberately reports missing evidence instead of inventing it.")
            st.download_button("Download HTML report", build_report(result), "satquery-report.html", "text/html", use_container_width=True)
else:
    st.info("Upload a remote-sensing image to begin.")
