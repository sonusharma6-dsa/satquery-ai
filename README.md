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

The app runs in transparent demo mode without model weights. For optional Hugging Face inference, install the model runtime and set these Streamlit Cloud secrets or local environment variables:

```text
SATQUERY_VQA_MODEL=dandelin/vilt-b32-finetuned-vqa
SATQUERY_CAPTION_MODEL=Salesforce/blip-image-captioning-base
```

The VQA and caption models are generic baselines. For the SIH domain-adaptation claim, replace them with your BigEarthNet-adapted checkpoint and document the adaptation protocol. The change-analysis path is available without model weights and generates a thresholded visual evidence map.

## Domain adaptation

Use `scripts/prepare_bigearthnet.py` to convert a downloaded BigEarthNet subset into JSONL records. Fine-tune RemoteCLIP or a small vision encoder with LoRA/land-cover multilabel supervision. BigEarthNet labels are not free-form VQA captions, so the adaptation claim should describe the label-to-text prompt construction and the resulting adapter weights.
