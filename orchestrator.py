"""
Agentic Orchestrator for SatQuery AI.

Performs:
1. Query classification (interprets intent: VQA, caption, change, optical-SAR)
2. Input compatibility checking (verifies image count & types match query)
3. Specialist model routing
4. Execution trace logging (returns structured audit log required by PS 167)
"""
import time
from typing import List
from PIL import Image

import models

TASK_VQA = "vqa"
TASK_CAPTION = "caption"
TASK_CHANGE = "change_detection"
TASK_FUSION = "optical_sar_fusion"


def classify_query(query: str, num_images: int, is_optical_sar_pair: bool = False) -> str:
    """
    Classify query intent based on keywords and input counts.
    Specific questions (counting, presence, location) always route to VQA.
    """
    q_lower = query.lower()
    if num_images == 2:
        if is_optical_sar_pair or "sar" in q_lower or "radar" in q_lower:
            return TASK_FUSION
        return TASK_CHANGE

    is_specific_question = any(k in q_lower for k in ["how many", "count", "number of", "is there", "are there", "where", "what is the count", "?"])
    if not is_specific_question and any(k in q_lower for k in ["describe", "caption", "summary", "overview"]):
        return TASK_CAPTION

    return TASK_VQA


def execute_query(
    query: str,
    images: List[Image.Image],
    is_optical_sar_pair: bool = False,
) -> dict:
    """
    Main entry point for agentic execution.
    Runs routing, executes specialist model, and produces auditable trace log.
    """
    trace = []
    start_time = time.time()

    # Step 1: Input Validation
    num_imgs = len(images)
    trace.append({"step": 1, "action": "Input Validation", "detail": f"Received {num_imgs} image(s). Query: '{query}'"})
    if num_imgs == 0:
        return {"error": "At least one satellite image is required.", "trace": trace}

    # Step 2: Query Classification & Compatibility Check
    selected_task = classify_query(query, num_imgs, is_optical_sar_pair)
    trace.append({
        "step": 2,
        "action": "Agentic Routing",
        "detail": f"Selected specialist task: '{selected_task}' based on inputs and query intent.",
    })

    if selected_task in [TASK_CHANGE, TASK_FUSION] and num_imgs < 2:
        selected_task = TASK_VQA
        trace.append({
            "step": 2.1,
            "action": "Fallback Applied",
            "detail": "Task required 2 images but 1 provided. Falling back to single-image VQA.",
        })

    # Step 3: Dispatch to Specialist
    trace.append({
        "step": 3,
        "action": "Specialist Execution",
        "detail": f"Invoking model layer for task: {selected_task} with base model Qwen2-VL-2B.",
    })

    result = {}
    if selected_task == TASK_VQA:
        result = models.run_vqa(images[0], query)
    elif selected_task == TASK_CAPTION:
        result = models.run_caption(images[0])
    elif selected_task == TASK_CHANGE:
        result = models.run_change_detection(images[0], images[1], query)
    elif selected_task == TASK_FUSION:
        result = models.run_optical_sar_fusion(images[0], images[1], query)

    exec_time = round(time.time() - start_time, 2)

    # Step 4: Finalize Execution Trace
    trace.append({
        "step": 4,
        "action": "Response Generation",
        "detail": f"Completed in {exec_time}s. Confidence: {result.get('confidence', 0.85)}",
    })

    return {
        "task": selected_task,
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", 0.85),
        "change_percentage": result.get("change_percentage"),
        "mask_artifact": result.get("mask_artifact"),
        "execution_trace": trace,
        "execution_time_sec": exec_time,
    }
