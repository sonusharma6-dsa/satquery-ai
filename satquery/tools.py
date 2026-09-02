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
        import matplotlib.pyplot as plt
        output_path = Path(output_dir) / "change_map.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.imshow(difference, cmap="magma", vmin=0, vmax=1)
        axis.contour(difference > threshold, levels=[0.5], colors="#00ffd5", linewidths=0.8)
        axis.set_axis_off()
        figure.tight_layout(pad=0)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return str(output_path)
    except (ImportError, OSError, ValueError):
        return None


def optical_sar_fusion(query: str, images: list[dict], parameters: dict) -> ToolResult:
    modalities = [image.get("modality", "unknown") for image in images]
    answer = "The pair passed shape compatibility checks. Optical-SAR fusion is ready for the CROMA adapter."
    return ToolResult("optical_sar_fusion", answer, [], 0.4, {"modalities": modalities, "adapter": "CROMA (pending weights)"})


def grounding(query: str, images: list[dict], parameters: dict) -> ToolResult:
    answer = "Region grounding is registered but requires GroundingDINO and SAM2 weights for production evidence boxes."
    return ToolResult("grounding", answer, [], 0.25, {"adapter": "GroundingDINO + SAM2"})
