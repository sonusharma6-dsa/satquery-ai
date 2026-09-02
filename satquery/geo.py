from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def read_asset(path: str, name: str | None = None) -> dict[str, Any]:
    file_path = Path(path)
    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported formats: GeoTIFF, TIFF, PNG, and JPEG.")

    metadata: dict[str, Any] = {}
    if extension in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError("Install Rasterio to read GeoTIFF files.") from exc
        with rasterio.open(file_path) as source:
            raw = source.read().astype("float32")
            metadata = {
                "crs": str(source.crs) if source.crs else None,
                "transform": str(source.transform),
                "width": source.width,
                "height": source.height,
                "bands": source.count,
            }
        array = np.moveaxis(raw, 0, -1)
    else:
        array = np.asarray(Image.open(file_path).convert("RGB")).astype("float32")
        array /= 255.0
        metadata = {
            "crs": None,
            "transform": None,
            "width": array.shape[1],
            "height": array.shape[0],
            "bands": array.shape[2],
        }

    array = np.nan_to_num(array)
    low = np.percentile(array, 2, axis=(0, 1), keepdims=True)
    high = np.percentile(array, 98, axis=(0, 1), keepdims=True)
    normalized = np.clip((array - low) / (high - low + 1e-6), 0, 1)
    return {"name": name or file_path.name, "path": str(file_path), "array": normalized, "extension": extension, **metadata}


def preview_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError("Expected an image with shape height x width x bands.")
    if array.shape[2] == 1:
        image = np.repeat(array, 3, axis=2)
    elif array.shape[2] >= 3:
        image = array[:, :, :3]
    else:
        image = np.repeat(array[:, :, :2], 2, axis=2)[:, :, :3]
    return np.clip(image, 0, 1)
