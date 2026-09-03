from __future__ import annotations

import re

from .schemas import AnalysisPlan
from .tools import change_analysis, grounding, optical_sar_fusion, scene_description, single_image_vqa

TOOL_REGISTRY = {
    "vqa": single_image_vqa,
    "captioning": scene_description,
    "grounding": grounding,
    "change_analysis": change_analysis,
    "optical_sar_fusion": optical_sar_fusion,
}


def plan_query(query: str, image_count: int) -> AnalysisPlan:
    """Demo planner; replace with structured JSON from an LLM in production."""
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    tasks: list[str] = []
    if image_count == 2 and words & {"change", "changed", "difference", "before", "after"}:
        tasks.append("change_analysis")
    if image_count == 2 and words & {"sar", "radar", "optical", "fusion", "combine"}:
        tasks.append("optical_sar_fusion")
    if words & {"where", "region", "locate", "highlight", "area"}:
        tasks.append("grounding")
    if words & {"describe", "caption", "scene", "visible"}:
        tasks.append("captioning")
    if not tasks:
        tasks.append("vqa")
    return AnalysisPlan(tasks, image_count, "unknown", "; ".join(tasks), {"change_threshold": 0.18})


def validate(plan: AnalysisPlan, images: list[dict]) -> None:
    if len(images) not in {1, 2}:
        raise ValueError("Upload one image or a compatible pair.")
    if len(images) != plan.required_images:
        raise ValueError(f"The selected analysis requires {plan.required_images} image(s).")
    if len(images) == 2:
        first, second = images
        if (first["width"], first["height"]) != (second["width"], second["height"]):
            raise ValueError("The pair must have matching width and height.")
        if plan.modality == "optical_sar" and {image.get("modality") for image in images} != {"optical", "sar"}:
            raise ValueError("Optical-SAR analysis requires one optical image and one SAR image.")
        if plan.modality == "bi_temporal" and not all(image.get("registered") for image in images):
            raise ValueError("Bi-temporal analysis requires co-registration confirmation for both images.")


def run_analysis(query: str, images: list[dict], parameters: dict | None = None) -> dict:
    if not query.strip():
        raise ValueError("Ask a question about the uploaded scene.")
    plan = plan_query(query, len(images))
    if len(images) == 2 and "optical_sar_fusion" in plan.tasks:
        plan.modality = "optical_sar"
    elif len(images) == 2 and "change_analysis" in plan.tasks:
        plan.modality = "bi_temporal"
    if parameters:
        plan.parameters.update(parameters)
    validate(plan, images)
    results = [TOOL_REGISTRY[task](query, images, plan.parameters) for task in plan.tasks]
    return {
        "answer": "\n\n".join(result.text for result in results),
        "evidence": [path for result in results for path in result.evidence_paths],
        "confidence": min((result.confidence for result in results), default=0),
        "plan": plan,
        "results": results,
    }
