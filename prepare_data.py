"""
Data Preprocessing Script for RSVQA Dataset.

Converts downloaded RSVQA JSON format into JSONL format required for
Qwen2-VL LoRA fine-tuning.

Usage:
    python prepare_data.py --rsvqa_dir ./RSVQA_LR --output train_data.jsonl --limit 3000

RSVQA Download: https://rsvqa.sylvainlobry.com/

NOTE: RSVQA's exact JSON field names vary slightly between dataset releases.
Check the printed keys the first time you run this against your actual
downloaded files, and adjust load_rsvqa() below if needed (e.g. "img_id"
vs "img_name", or a different nesting under "answers").
"""
import argparse
import json
import os


def load_rsvqa(rsvqa_dir: str, limit: int = 3000) -> list:
    """
    Parse RSVQA Low-Resolution JSON files.
    Standard layout:
        RSVQA_LR/
            Images_LR/ (contains .png/.tif files)
            Questions_LR.json
            Answers_LR.json
    """
    q_file = os.path.join(rsvqa_dir, "Questions_LR.json")
    a_file = os.path.join(rsvqa_dir, "Answers_LR.json")
    if not os.path.exists(q_file):
        q_file = os.path.join(rsvqa_dir, "LR_split_train_questions.json")
        a_file = os.path.join(rsvqa_dir, "LR_split_train_answers.json")
    img_dir = os.path.join(rsvqa_dir, "Images_LR")

    if not os.path.exists(q_file):
        raise FileNotFoundError(f"Missing {q_file}. Download RSVQA from https://rsvqa.sylvainlobry.com/")

    with open(q_file, "r") as f:
        questions_data = json.load(f)["questions"]
    with open(a_file, "r") as f:
        answers_data = json.load(f)["answers"]

    ans_map = {
        a.get("question_id", a.get("id")): a["answer"]
        for a in answers_data
        if a.get("active", True) and "answer" in a
    }

    converted = []
    for q in questions_data[:limit]:
        q_text = q["question"]
        answer_ids = q.get("answers_ids", [q.get("answer_id")])
        ans_id = answer_ids[0] if answer_ids else None
        img_id = q.get("img_id")
        img_name = q.get("img_name")
        if not img_name:
            img_name = next(
                (
                    name
                    for name in (f"{img_id}.tif", f"{img_id}.png")
                    if os.path.exists(os.path.join(img_dir, name))
                ),
                f"{img_id}.png",
            )
        img_path = os.path.join(img_dir, img_name)
        answer_key = q.get("id", ans_id)
        if answer_key in ans_map:
            ans_text = str(ans_map[answer_key])
            converted.append({
                "image": img_path,
                "query": f"You are SatQuery AI. Answer this remote sensing question: {q_text}",
                "response": ans_text,
            })
    return converted


def main():
    parser = argparse.ArgumentParser(description="Prepare RSVQA data for LoRA fine-tuning.")
    parser.add_argument("--rsvqa_dir", type=str, required=True, help="Path to unzipped RSVQA folder")
    parser.add_argument("--output", type=str, default="train_data.jsonl", help="Output JSONL file path")
    parser.add_argument("--limit", type=int, default=3000, help="Max examples to convert (default: 3000 for fast training)")
    args = parser.parse_args()

    print(f"Loading RSVQA from {args.rsvqa_dir}...")
    records = load_rsvqa(args.rsvqa_dir, args.limit)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[SUCCESS] Successfully converted {len(records)} examples -> {args.output}")


if __name__ == "__main__":
    main()
