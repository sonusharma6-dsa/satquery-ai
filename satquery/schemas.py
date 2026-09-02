from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImageAsset:
    name: str
    path: str
    extension: str
    width: int
    height: int
    bands: int
    modality: str = "unknown"
    crs: str | None = None
    transform: str | None = None
    array: Any = None


@dataclass
class AnalysisPlan:
    tasks: list[str]
    required_images: int
    modality: str
    intent: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    task: str
    text: str
    evidence_paths: list[str]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
