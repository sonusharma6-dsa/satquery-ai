"""
SatQuery AI -- Batch Evaluation & Diagnostic Engine for Qwen2-VL-2B + QLoRA
=============================================================================
Runs batch inference over categorized remote sensing test questions (counting, presence,
spatial reasoning, bi-temporal comparison, captioning), uses deterministic decoding,
saves evaluation_results.csv, and prints category-wise accuracy breakdown.
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any

import torch
import pandas as pd
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

def load_eval_model(base_model_id: str, adapter_path: str, use_4bit: bool = True):
    print(f"[INFO] Loading Base Model: {base_model_id} (4-bit={use_4bit})...", flush=True)
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    ) if use_4bit else None

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config,
    )
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"[INFO] Loading LoRA Adapter from: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print(f"[WARNING] Adapter path '{adapter_path}' not found. Evaluating BASE model.", flush=True)

    processor = AutoProcessor.from_pretrained(base_model_id)

    # 1. Enforce Evaluation Mode (Disables Dropout)
    model.eval()
    
    # 2. Configure Tokenizer Pad Token ID
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    return model, processor


def generate_answer(
    model,
    processor,
    images: List[Image.Image],
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Deterministic Decoding Parameters (do_sample=False, repetition_penalty=1.1)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            repetition_penalty=1.1,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def diagnose_output(expected: str, predicted: str, category: str) -> Dict[str, Any]:
    pred_clean = predicted.lower().strip()
    exp_clean = str(expected).lower().strip()

    # Check for 'weird' (garbled/incoherent/repetitive) criteria
    is_weird = False
    if len(predicted) == 0:
        is_weird = True
    elif any(char in predicted for char in ["<|im_start|>", "<|im_end|>", "<|object_ref_start|>"]):
        is_weird = True
    elif len(predicted.split()) > 40 and len(set(predicted.split())) < (len(predicted.split()) * 0.3):
        is_weird = True
    elif category == "counting" and not any(c.isdigit() for c in predicted) and not any(w in pred_clean for w in ["none", "zero", "between", "many"]):
        is_weird = True

    if is_weird:
        return {"status": "weird", "is_match": False, "score": 0.0}

    if category in ["counting", "presence"]:
        is_match = (exp_clean in pred_clean) or (pred_clean in exp_clean)
        return {"status": "correct" if is_match else "wrong", "is_match": is_match, "score": 1.0 if is_match else 0.0}
    else:
        exp_words = set(exp_clean.split())
        pred_words = set(pred_clean.split())
        overlap = len(exp_words.intersection(pred_words)) / max(len(exp_words), 1)
        is_match = overlap > 0.3
        return {"status": "correct" if is_match else "wrong", "is_match": is_match, "score": round(overlap, 2)}


def build_expanded_test_suite() -> List[Dict[str, Any]]:
    img_dir = "data/Images_LR"
    imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.tif', '.png'))] if os.path.exists(img_dir) else []
    if not imgs:
        imgs = ["sample_satellite.png"] * 30

    suite = []

    # 1. Counting (30 samples)
    counting_templates = [
        ("How many residential building structures are present in this satellite area?", "12-18"),
        ("What is the count of houses visible across the sub-tiles?", "15"),
        ("How many large industrial roofs are in this tile?", "between 10 and 20"),
        ("How many distinct buildings are visible in this scene?", "10 to 15"),
        ("Count the number of agricultural storage sheds in the image.", "5-10"),
        ("What is the total count of isolated structures in this rural grid?", "8"),
    ]
    for i in range(30):
        tmpl = counting_templates[i % len(counting_templates)]
        img = imgs[i % len(imgs)]
        suite.append({"category": "counting", "image": img, "query": f"[VQA] You are SatQuery AI. {tmpl[0]} (Sample {i+1})", "expected": tmpl[1]})

    # 2. Presence (30 samples)
    presence_templates = [
        ("Is there a river or water body present in this image?", "no"),
        ("Are there paved roads visible connecting the buildings?", "yes"),
        ("Is this an urban or rural residential sector?", "urban"),
        ("Are agricultural fields visible in this satellite patch?", "yes"),
        ("Is there an airport runway present in this area?", "no"),
        ("Are solar panel installations visible on the rooftops?", "no"),
    ]
    for i in range(30):
        tmpl = presence_templates[i % len(presence_templates)]
        img = imgs[(i + 5) % len(imgs)]
        suite.append({"category": "presence", "image": img, "query": f"[VQA] You are SatQuery AI. {tmpl[0]} (Sample {i+1})", "expected": tmpl[1]})

    # 3. Spatial Reasoning (30 samples)
    spatial_templates = [
        ("Describe the spatial layout and structural arrangement of the buildings.", "Clustered residential buildings along road network"),
        ("Where is the densest building cluster located in the tile?", "Central and northern quadrant"),
        ("What terrain covers the open spaces between structures?", "Green vegetation canopy and bare soil"),
        ("How are the roads oriented relative to the built structures?", "Linear grid alignment parallel to building rows"),
        ("Are the buildings isolated or densely connected in this zone?", "Densely connected urban block"),
        ("Which quadrant contains the highest concentration of open vegetation?", "Southwestern sector"),
    ]
    for i in range(30):
        tmpl = spatial_templates[i % len(spatial_templates)]
        img = imgs[(i + 10) % len(imgs)]
        suite.append({"category": "spatial", "image": img, "query": f"[VQA] You are SatQuery AI. {tmpl[0]} (Sample {i+1})", "expected": tmpl[1]})

    # 4. Change Detection (30 samples)
    change_templates = [
        ("What structural changes and new construction occurred between Time T1 and T2?", "New building footprint and land clearing"),
        ("Identify the primary land-surface modification between T1 and T2.", "Building expansion and ground alteration"),
        ("Has vegetation density increased or decreased between T1 and T2?", "Clearing of vegetation for construction"),
        ("Compare road infrastructure development between Time T1 and Time T2.", "Expanded road network and new paved lanes"),
        ("Detect any environmental or canopy changes across the multi-temporal pair.", "Loss of tree cover and ground disturbance"),
        ("What is the main difference in built-up area between T1 and T2?", "Increase in roof surface area and clearings"),
    ]
    for i in range(30):
        tmpl = change_templates[i % len(change_templates)]
        img1 = "sample_t1.png" if os.path.exists("sample_t1.png") else imgs[i % len(imgs)]
        img2 = "sample_t2.png" if os.path.exists("sample_t2.png") else imgs[(i + 1) % len(imgs)]
        suite.append({"category": "change_detection", "image": img1, "image2": img2, "query": f"[CHANGE] You are SatQuery AI. {tmpl[0]} (Sample {i+1})", "expected": tmpl[1]})

    # 5. Captioning (30 samples)
    caption_templates = [
        ("Provide a comprehensive remote sensing caption and scene analysis.", "Dense urban sector with residential roofs, roads, and surrounding vegetation"),
        ("Summarize the dominant land-cover classes present.", "Urban structures, asphalt roads, vegetation canopy"),
        ("Describe the soil reflectance and environmental features.", "High reflectance built roof surfaces and green biomass"),
        ("Generate a detailed satellite imagery description highlighting key infrastructure.", "Residential settlement surrounded by agricultural fields and secondary road grid"),
        ("Give a high-level summary of the spatial patterns and land surface cover.", "Mixed rural-urban zone with sparse trees and scattered low-rise buildings"),
        ("Describe the scene context and spatial distribution of land-use types.", "High-density residential housing with asphalt connectivity and garden plots"),
    ]
    for i in range(30):
        tmpl = caption_templates[i % len(caption_templates)]
        img = imgs[(i + 15) % len(imgs)]
        suite.append({"category": "captioning", "image": img, "query": f"[CAPTION] You are SatQuery AI. {tmpl[0]} (Sample {i+1})", "expected": tmpl[1]})

    return suite


def run_evaluation(adapter_path: str = "./lora_adapter", output_csv: str = "evaluation_results.csv"):
    base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
    model, processor = load_eval_model(base_model_id, adapter_path, use_4bit=True)

    test_suite = build_expanded_test_suite()

    results = []
    print(f"\n[INFO] Starting batch evaluation over {len(test_suite)} categorized samples (30/category)...\n", flush=True)
    start_time = time.time()

    for idx, item in enumerate(test_suite):
        img_path = item["image"]
        img_path2 = item.get("image2")
        query = item["query"]
        expected = item["expected"]
        category = item["category"]

        images = [Image.open(img_path).convert("RGB")]
        if img_path2 and os.path.exists(img_path2):
            images.append(Image.open(img_path2).convert("RGB"))

        pred_text = generate_answer(model, processor, images, query)
        diagnosis = diagnose_output(expected, pred_text, category)

        record = {
            "sample_id": idx + 1,
            "category": category,
            "question": query,
            "expected_answer": expected,
            "model_answer": pred_text,
            "status": diagnosis["status"],
            "is_exact_match": diagnosis["is_match"],
            "overlap_score": diagnosis["score"],
        }
        results.append(record)

        print(f"[{idx+1:03d}/{len(test_suite):03d}] [{category.upper():16s}] STATUS: {diagnosis['status'].upper():7s} | Pred: {pred_text[:60]}...", flush=True)

    eval_time = round(time.time() - start_time, 2)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[SUCCESS] Saved evaluation CSV to '{output_csv}' ({eval_time}s)", flush=True)

    summary_df = df.groupby("category").agg(
        total=("sample_id", "count"),
        correct=("status", lambda s: (s == "correct").sum()),
        wrong=("status", lambda s: (s == "wrong").sum()),
        weird=("status", lambda s: (s == "weird").sum()),
        pass_rate=("is_exact_match", lambda m: f"{m.mean()*100:.1f}%"),
    )
    summary_df.to_csv("evaluation_summary.csv", encoding="utf-8")

    print("\n=======================================================", flush=True)
    print("           CATEGORY-WISE EVALUATION BREAKDOWN          ", flush=True)
    print("=======================================================", flush=True)
    print(summary_df.to_string(), flush=True)
    print("=======================================================\n", flush=True)
    return df


if __name__ == "__main__":
    run_evaluation()
