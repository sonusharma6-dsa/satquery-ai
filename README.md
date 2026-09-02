# SatQuery AI

A transparent, multimodal remote-sensing assistant for SIH26167. The demo supports GeoTIFF/TIFF/PNG/JPEG upload, single-image analysis, bi-temporal comparison, optical-SAR compatibility checks, visual evidence overlays, confidence estimates, an execution trace, and downloadable HTML reports.

## Run

Use Python 3.11 or 3.12 (PyTorch and Rasterio are not consistently available for Python 3.14):

```powershell
py -3.11 -m venv .venv-satquery
.\.venv-satquery\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Create a GitHub repository containing this folder, then choose `app.py` as the
main file in Streamlit Community Cloud. The repository must include
`requirements.txt` and `.python-version`. Do not commit `.venv`, downloaded
weights, secrets, or raw datasets.

The app attempts real Hugging Face inference by default. The first VQA or caption request downloads model weights, so the first request can take several minutes. Override the defaults with these Streamlit Cloud secrets or local environment variables:

```text
SATQUERY_VQA_MODEL=dandelin/vilt-b32-finetuned-vqa
SATQUERY_CAPTION_MODEL=Salesforce/blip-image-captioning-base
```

The VQA and caption models are generic baselines. For the SIH domain-adaptation claim, replace them with your BigEarthNet-adapted checkpoint and document the adaptation protocol. The change-analysis path is available without model weights and generates a thresholded visual evidence map.

## SIH domain adaptation

After preparing a BigEarthNet JSONL manifest, run this on Colab/Kaggle:

```bash
python scripts/train_bigearthnet_adapter.py data/processed/bigearthnet.jsonl weights/bigearthnet-adapter --epochs 2
```

The script freezes a CLIP vision encoder and trains a multilabel land-cover head using BigEarthNet labels. It saves `bigearthnet_head.pt` and an adapter card. Only after this succeeds should the demo be described as domain-adapted. The ISRO Cartosat-2S/RISAT pair can then be supplied through the same GeoTIFF upload path, with matching dimensions and registration metadata.

Set `SATQUERY_REMOTE_ADAPTER` to the adapter directory when it has been trained. The execution summary reports this value so judges can distinguish an adapted run from a generic baseline or transparent fallback.

## Domain adaptation

Use `scripts/prepare_bigearthnet.py` to convert a downloaded BigEarthNet subset into JSONL records. Fine-tune RemoteCLIP or a small vision encoder with LoRA/land-cover multilabel supervision. BigEarthNet labels are not free-form VQA captions, so the adaptation claim should describe the label-to-text prompt construction and the resulting adapter weights.
