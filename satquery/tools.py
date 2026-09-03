from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .schemas import ToolResult


def _question_words(query: str) -> list[str]:
    return [word.strip(".,?!").lower() for word in query.split()]


@lru_cache(maxsize=2)
def _load_pipeline(task: str, model_id: str):
    """Load optional Hugging Face models only when an operator enables them."""
    from transformers import pipeline
    return pipeline(task, model=model_id, device=-1)


def _model_answer(task: str, model_id: str, query: str, image: dict) -> str | None:
    try:
        model = _load_pipeline(task, model_id)
        pil_image = Image.fromarray((image["array"][:, :, :3] * 255).astype("uint8"))
        if task == "visual-question-answering":
            output = model(image=pil_image, question=query)
            return str(output[0].get("answer", output[0]))
        output = model(pil_image)
        return str(output[0].get("generated_text", output[0]))
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return f"MODEL_UNAVAILABLE: {error}"


def single_image_vqa(query: str, images: list[dict], parameters: dict) -> ToolResult:
    image = images[0]
    bands = image["bands"]
    adapter = os.getenv("SATQUERY_REMOTE_ADAPTER", "not_configured")
    model_id = os.getenv("SATQUERY_VQA_MODEL", "dandelin/vilt-b32-finetuned-vqa").strip()
    if model_id:
        answer = _model_answer("visual-question-answering", model_id, query, image)
        if answer and not answer.startswith("MODEL_UNAVAILABLE:"):
            return ToolResult("vqa", answer, [], 0.68, {"mode": "huggingface", "model": model_id, "remote_adapter": adapter, "bands": bands})
    answer = (
        "The uploaded scene is available for analysis, but no production VLM weights "
        "are installed yet. The deterministic preview confirms a "
        f"{bands}-band {image['extension'].upper()} image."
    )
    return ToolResult("vqa", answer, [], 0.35, {"mode": "transparent_demo", "remote_adapter": adapter, "bands": bands})


def scene_description(query: str, images: list[dict], parameters: dict) -> ToolResult:
    image = images[0]
    adapter = os.getenv("SATQUERY_REMOTE_ADAPTER", "not_configured")
    model_id = os.getenv("SATQUERY_CAPTION_MODEL", "Salesforce/blip-image-captioning-base").strip()
    if model_id:
        answer = _model_answer("image-to-text", model_id, query, image)
        if answer and not answer.startswith("MODEL_UNAVAILABLE:"):
            return ToolResult("captioning", answer, [], 0.62, {"mode": "huggingface", "model": model_id, "remote_adapter": adapter})
    array = image["array"]
    mean_intensity = float(np.mean(array))
    scene_hint = "bright reflective surfaces" if mean_intensity > 0.55 else "lower-reflectance surfaces"
    answer = f"The image contains {scene_hint}; detailed semantic labels require the adapted VLM adapter."
    return ToolResult("captioning", answer, [], 0.32, {"mode": "transparent_demo", "remote_adapter": adapter, "mean_normalized_intensity": round(mean_intensity, 3)})


def change_analysis(query: str, images: list[dict], parameters: dict) -> ToolResult:
    first, second = images
    first_array = first["array"]
    second_array = second["array"]
    common_bands = min(first_array.shape[2], second_array.shape[2])
    first_array = first_array[:, :, :common_bands]
    second_array = second_array[:, :, :common_bands]
    if first_array.shape[:2] != second_array.shape[:2]:
        raise ValueError("Temporal images must have identical dimensions after registration.")
    difference = np.mean(np.abs(second_array - first_array), axis=2)
    threshold = float(parameters.get("change_threshold", 0.18))
    changed_fraction = float(np.mean(difference > threshold))
    evidence_path = _save_change_map(difference, threshold, parameters.get("output_dir"))
    answer = f"The preliminary change detector flags {changed_fraction:.1%} of pixels above the configured threshold."
    return ToolResult("change_analysis", answer, [evidence_path] if evidence_path else [], min(0.85, 0.4 + changed_fraction), {"threshold": threshold, "changed_fraction": changed_fraction, "evidence": evidence_path or "not_saved"})


def _save_change_map(difference: np.ndarray, threshold: float, output_dir: str | None) -> str | None:
    if not output_dir:
        return None
    try:
        output_path = Path(output_dir) / "change_map.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        intensity = np.clip(difference, 0, 1)
        heatmap = np.stack(
            [255 * intensity, 80 * (1 - intensity), 180 * (1 - intensity)], axis=2
        ).astype("uint8")
        evidence = Image.fromarray(heatmap, mode="RGB")
        changed = difference > threshold
        boundary = changed & ~(
            np.roll(changed, 1, axis=0)
            & np.roll(changed, -1, axis=0)
            & np.roll(changed, 1, axis=1)
            & np.roll(changed, -1, axis=1)
        )
        overlay = Image.new("RGBA", evidence.size, (0, 0, 0, 0))
        overlay_pixels = np.asarray(overlay).copy()
        overlay_pixels[boundary] = (0, 255, 213, 255)
        evidence = Image.alpha_composite(evidence.convert("RGBA"), Image.fromarray(overlay_pixels, mode="RGBA"))
        evidence.convert("RGB").save(output_path)
        return str(output_path)
    except (OSError, ValueError, TypeError):
        return None


def optical_sar_fusion(query: str, images: list[dict], parameters: dict) -> ToolResult:
    modalities = [image.get("modality", "unknown") for image in images]
    optical = next((image for image in images if image.get("modality") == "optical"), None)
    sar = next((image for image in images if image.get("modality") == "sar"), None)
    if optical is None or sar is None:
        answer = "The pair passed shape checks, but optical-SAR fusion requires explicit modality labels."
        return ToolResult("optical_sar_fusion", answer, [], 0.2, {"modalities": modalities, "adapter": "CROMA"})
    optical_signal = float(np.mean(optical["array"]))
    sar_signal = float(np.mean(sar["array"]))
    fused_signal = (optical_signal + sar_signal) / 2
    answer = f"The registered pair combines optical reflectance ({optical_signal:.3f}) with SAR backscatter ({sar_signal:.3f}); fused normalized signal is {fused_signal:.3f}."
    return ToolResult("optical_sar_fusion", answer, [], 0.56, {"modalities": modalities, "adapter": "CROMA-compatible statistical fusion", "optical_signal": round(optical_signal, 3), "sar_signal": round(sar_signal, 3), "fused_signal": round(fused_signal, 3)})


def grounding(query: str, images: list[dict], parameters: dict) -> ToolResult:
    image = images[0]
    model_id = os.getenv("SATQUERY_GROUNDING_MODEL", "IDEA-Research/grounding-dino-tiny").strip()
    labels = _grounding_labels(query)
    try:
        detector = _load_pipeline("zero-shot-object-detection", model_id)
        pil_image = Image.fromarray((image["array"][:, :, :3] * 255).astype("uint8"))
        detections = detector(pil_image, candidate_labels=labels)
        evidence_path = _save_grounding_map(pil_image, detections, parameters.get("output_dir"))
        confident = [item for item in detections if item.get("score", 0) >= 0.25]
        if confident:
            found = ", ".join(f"{item['label']} ({item['score']:.0%})" for item in confident[:5])
            answer = f"The grounding model located: {found}."
        else:
            answer = "The grounding model did not find a candidate region above its confidence threshold."
        return ToolResult("grounding", answer, [evidence_path] if evidence_path else [], 0.62 if confident else 0.35, {"mode": "huggingface", "model": model_id, "candidate_labels": labels, "detections": len(confident)})
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        answer = "Region grounding is unavailable until the Grounding DINO model can be downloaded; no boxes were invented."
        return ToolResult("grounding", answer, [], 0.25, {"mode": "transparent_demo", "adapter": model_id, "error": str(error)})


def _grounding_labels(query: str) -> list[str]:
    known_labels = ["building", "road", "water", "forest", "farmland", "vehicle", "ship", "airplane"]
    words = {word.strip(".,?!").lower() for word in query.split()}
    selected = [label for label in known_labels if label in words or f"{label}s" in words]
    return selected or known_labels


def _save_grounding_map(image: Image.Image, detections: list[dict], output_dir: str | None) -> str | None:
    if not output_dir:
        return None
    try:
        from PIL import ImageDraw
        output_path = Path(output_dir) / "grounding_boxes.png"
        evidence = image.copy()
        draw = ImageDraw.Draw(evidence)
        for item in detections:
            if item.get("score", 0) < 0.25:
                continue
            box = item["box"]
            coordinates = (box["xmin"], box["ymin"], box["xmax"], box["ymax"])
            draw.rectangle(coordinates, outline="#00ffd5", width=3)
            draw.text((box["xmin"] + 3, box["ymin"] + 3), f"{item['label']} {item['score']:.0%}", fill="#00ffd5")
        evidence.save(output_path)
        return str(output_path)
    except (OSError, ValueError):
        return None
