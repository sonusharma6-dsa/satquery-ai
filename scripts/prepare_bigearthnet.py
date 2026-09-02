"""Convert a BigEarthNet subset's metadata into a simple JSONL manifest.

Run this against files downloaded from the official BigEarthNet source. The
script intentionally does not download data automatically because access,
licenses, and archive layouts vary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="BigEarthNet subset directory")
    parser.add_argument("output", type=Path, help="Output JSONL manifest")
    args = parser.parse_args()
    records = []
    for label_file in args.root.rglob("*_labels_metadata.json"):
        metadata = json.loads(label_file.read_text(encoding="utf-8"))
        patch_dir = label_file.parent
        records.append({
            "image": str(patch_dir),
            "labels": metadata.get("labels", metadata.get("original_labels", [])),
            "task": "remote_sensing_adaptation",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
