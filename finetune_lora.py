"""
LoRA/QLoRA Fine-Tuning Script for Qwen2-VL-2B on Remote Sensing VQA (RSVQA).

Uses 4-bit quantization (QLoRA) by default so it fits on smaller-VRAM GPUs
like an RTX 3050 (4GB or 8GB). Set USE_4BIT = False below if you have a
bigger GPU or are running on Colab's free T4 (16GB) and want full fp16 LoRA.

Usage:
    python finetune_lora.py --train_data train_data.jsonl --output_dir ./lora_adapter --epochs 2
"""
import argparse
import json

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from torch.optim import AdamW

USE_4BIT = True


class RSVQAJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = item["image"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback dummy image if file missing during initial testing
            image = Image.new("RGB", (256, 256), color="green")
        return {"image": image, "query": item["query"], "response": item["response"]}


def train_lora(train_data_path: str, output_dir: str, epochs: int = 2, batch_size: int = 2, lr: float = 2e-4):
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        print("Loading base model in 4-bit (QLoRA) mode...")
    else:
        print("Loading base model in fp16...")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = RSVQAJsonlDataset(train_data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda items: {
            key: [item[key] for item in items]
            for key in items[0]
        },
    )
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    print(f"Starting LoRA fine-tuning for {epochs} epochs ({len(dataset)} samples)...")

    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            images = batch["image"]
            queries = batch["query"]
            responses = batch["response"]

            batch_loss = 0.0
            for img, q, r in zip(images, queries, responses):
                messages = [
                    {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]},
                    {"role": "assistant", "content": [{"type": "text", "text": r}]},
                ]
                text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                inputs["labels"] = inputs.input_ids.clone()
                outputs = model(**inputs)
                loss = outputs.loss / batch_size
                loss.backward()
                batch_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss

            if (step + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(dataloader)}] Loss: {batch_loss:.4f}")

        print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss / len(dataloader):.4f}")

    print(f"Saving LoRA adapter checkpoint to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("[SUCCESS] Fine-tuning complete! Set ADAPTER_PATH in models.py to use this adapter.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to train_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="./lora_adapter", help="Directory to save LoRA weights")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    train_lora(args.train_data, args.output_dir, args.epochs)
