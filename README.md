# SatQuery AI -- Quick Start Guide (ISRO PS 167)

Prototype covering all mandatory tasks for **SIH26167**:
1. Single-image VQA
2. Captioning
3. Bi-temporal Change Detection
4. Cross-modal Optical + SAR Fusion
5. Auditable Execution Trace Logging (agentic orchestrator)

---

## 💻 Running Locally (your RTX 3050)

The code defaults to **4-bit quantization (QLoRA)**, which fits comfortably
on a 4GB card and easily on 8GB.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

**Windows note:** `bitsandbytes` (needed for 4-bit) can be finicky on native
Windows. If `pip install bitsandbytes` fails or errors appear at runtime:
- Easiest fix: use **WSL2** (Windows Subsystem for Linux) and run everything from there.
- Or fall back to Google Colab (free T4 GPU, already Linux, no install issues).

## 🚀 Running on Free Google Colab (fallback / demo backup)

1. Open a new Colab notebook, set Runtime to **GPU (T4)**.
2. Upload `app.py`, `models.py`, `orchestrator.py`, `requirements.txt`.
3. In a cell:
```bash
!pip install -r requirements.txt
!pip install pyngrok
!streamlit run app.py & npx localtunnel --port 8501
```
4. Click the printed `localtunnel.me` link to open the web UI.

## 🎯 Datasets for Fine-Tuning

- BigEarthNet (domain adaptation): https://bigearth.net/
- RSVQA (VQA benchmark): https://rsvqa.sylvainlobry.com/
- VRSBench (grounding/captioning): https://github.com/csu-mrs/VRSBench

## 🔧 Fine-Tuning Pipeline

```bash
pip install peft bitsandbytes
python prepare_data.py --rsvqa_dir ./RSVQA_LR --output train_data.jsonl --limit 3000
# Smoke test first, on ~50 lines, before running the full set
python finetune_lora.py --train_data smoke_test.jsonl --epochs 1 --output_dir ./lora_smoke
# Full run once the smoke test's loss is clearly decreasing
python finetune_lora.py --train_data train_data.jsonl --epochs 2 --output_dir ./lora_adapter
```

After training, set `ADAPTER_PATH = "./lora_adapter"` at the top of
`models.py` and restart the app to use the fine-tuned weights.

## What's NOT done yet
- Actual fine-tuning run (this is a real training script, but you need to
  download RSVQA and run it — it hasn't been executed against real data yet).
- Real optical-SAR co-registration — the fusion task currently trusts that
  the two uploaded images are already aligned to the same area.
- Grounding (bounding-box localization) — only captioning is implemented
  as the second single-image task; add grounding if you have extra time.
