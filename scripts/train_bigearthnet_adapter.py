"""Train a lightweight BigEarthNet land-cover adapter.

This is the domain-adaptation artifact used by SatQuery AI. It adapts a
pretrained CLIP image encoder to BigEarthNet multilabel land-cover tags. The
output directory can later be supplied as SATQUERY_REMOTE_ENCODER. Download
BigEarthNet separately from its official source and create a JSONL manifest
with prepare_bigearthnet.py before running this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor


class BigEarthNetDataset(Dataset):
    def __init__(self, records, processor, classes):
        self.records = records
        self.processor = processor
        self.classes = classes
        self.encoder = {name: index for index, name in enumerate(classes)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = next(Path(record["image"]).glob("*.jpg"), None)
        if image_path is None:
            raise FileNotFoundError(f"No preview JPEG found in {record['image']}")
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        target = torch.zeros(len(self.classes), dtype=torch.float32)
        for label in record.get("labels", []):
            if label in self.encoder:
                target[self.encoder[label]] = 1
        return pixel_values, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = sorted({label for record in records for label in record.get("labels", [])})
    if not records or not labels:
        raise ValueError("Manifest must contain image records with BigEarthNet labels.")

    processor = CLIPProcessor.from_pretrained(args.model)
    encoder = CLIPModel.from_pretrained(args.model).vision_model
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    hidden_size = encoder.config.hidden_size
    head = nn.Linear(hidden_size, len(labels))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)
    head.to(device)
    loader = DataLoader(BigEarthNetDataset(records, processor, labels), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4)
    criterion = nn.BCEWithLogitsLoss()

    head.train()
    for epoch in range(args.epochs):
        losses = []
        for pixels, targets in loader:
            pixels, targets = pixels.to(device), targets.to(device)
            with torch.no_grad():
                features = encoder(pixel_values=pixels).pooler_output
                features = features / features.norm(dim=-1, keepdim=True)
            loss = criterion(head(features), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch={epoch + 1} loss={np.mean(losses):.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "labels": labels, "base_model": args.model}, args.output / "bigearthnet_head.pt")
    (args.output / "adapter_card.json").write_text(json.dumps({"dataset": "BigEarthNet", "task": "multilabel land-cover adaptation", "base_model": args.model, "labels": labels}, indent=2), encoding="utf-8")
    print(f"Saved BigEarthNet adapter to {args.output}")


if __name__ == "__main__":
    main()