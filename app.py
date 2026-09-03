from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from satquery.controller import run_analysis
from satquery.geo import read_asset, preview_rgb
from satquery.report import build_report

st.set_page_config(page_title="SatQuery AI", page_icon="🛰️", layout="wide")
st.markdown(
    """
    <style>
    .satquery-kicker { color: #087f5b; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
    .satquery-title { color: #17212b; font-size: 2.4rem; font-weight: 800; line-height: 1.05; margin: 0.15rem 0 0.35rem; }
    .satquery-subtitle { color: #52606d; font-size: 1.05rem; margin-bottom: 1.5rem; }
    [data-testid="stMetric"] { background: #f3f7f5; border-left: 4px solid #087f5b; padding: 0.8rem 1rem; }
    </style>
    <div class="satquery-kicker">SIH26167 | Explainable geospatial intelligence</div>
    <div class="satquery-title">SatQuery AI</div>
    <div class="satquery-subtitle">Evidence-grounded remote-sensing analysis with an auditable execution trace.</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Analysis settings")
    st.info("Accepted: GeoTIFF, TIFF, PNG, JPEG")
    threshold = st.slider("Change threshold", 0.05, 0.60, 0.18, 0.01)
    st.caption("This threshold is recorded in the audit report.")

uploads = st.file_uploader("Upload one image or a registered pair", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=True)
query = st.text_area("Ask SatQuery AI", placeholder="What changed between these two images?", height=90)

if uploads:
    if len(uploads) > 2:
        st.error("Upload at most two images: one scene or one registered pair.")
        st.stop()
    input_signature = hashlib.sha256(
        b"".join(upload.getvalue() for upload in uploads[:2])
        + query.strip().encode("utf-8")
        + str(threshold).encode("ascii")
    ).hexdigest()
    if st.session_state.get("input_signature") != input_signature:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_report", None)
        st.session_state["input_signature"] = input_signature

    columns = st.columns(min(len(uploads), 2))
    image_records = []
    temporary_directory = st.session_state.setdefault("analysis_output_dir", tempfile.mkdtemp(prefix="satquery-"))
    for index, upload in enumerate(uploads[:2]):
        saved_path = Path(temporary_directory) / f"input-{index}{Path(upload.name).suffix.lower()}"
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

    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    if analyze_clicked or st.session_state.get("analysis_result"):
        if analyze_clicked:
            if not query.strip():
                st.warning("Enter a question first.")
                st.stop()
            try:
                result = run_analysis(query, image_records, {"change_threshold": threshold, "output_dir": temporary_directory})
                st.session_state["analysis_result"] = result
                st.session_state["analysis_report"] = build_report(result)
            except Exception as error:
                st.error(str(error))
                st.stop()
        else:
            result = st.session_state["analysis_result"]

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
        evidence_images = [item.metadata.get("evidence") for item in result["results"] if item.metadata.get("evidence", "").endswith(".png") and Path(item.metadata["evidence"]).exists()]
        if evidence_images:
            for evidence_image in evidence_images:
                st.image(evidence_image, caption="Change intensity map with threshold contour", use_container_width=True)
        else:
            st.info("No visual evidence was produced for this task. The system deliberately reports missing evidence instead of inventing it.")
        st.download_button("Download HTML report", st.session_state.get("analysis_report", build_report(result)), "satquery-report.html", "text/html", use_container_width=True)
else:
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_report", None)
    st.info("Upload a remote-sensing image to begin.")
