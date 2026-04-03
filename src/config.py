"""Configuration dataclasses for the training audit."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LEXICON_DIR = DATA_DIR / "lexicons"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Religious traditions to classify, in priority order
TRADITIONS = ["christian", "islamic", "jewish", "hindu", "buddhist"]

# Known dataset configurations
DATASET_REGISTRY = {
    "pile": {
        "hf_path": "mit-han-lab/pile-val-backup",
        "fallback_path": "ola13/small-the_pile",
        "subset": None,
        "split": "validation",
        "text_field": "text",
        "meta_field": "meta",
        "source_subfield": "pile_set_name",
        "total_tokens": 825_000_000_000,
        "description": "The Pile (EleutherAI, ~825B tokens)",
    },
    "dolma": {
        "hf_path": "allenai/dolma",
        "fallback_path": None,
        "subset": "v1_7",
        "split": "train",
        "text_field": "text",
        "meta_field": "metadata",
        "source_subfield": "source",
        "total_tokens": 3_000_000_000_000,
        "description": "Dolma v1.7 (AI2, ~3T tokens)",
    },
    "fineweb": {
        "hf_path": "HuggingFaceFW/fineweb",
        "fallback_path": None,
        "subset": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "meta_field": None,
        "source_subfield": None,
        "total_tokens": 15_000_000_000_000,
        "description": "FineWeb (HuggingFace, ~15T tokens)",
    },
}


@dataclass
class DatasetConfig:
    """Configuration for a single HuggingFace dataset."""
    name: str
    hf_path: str
    fallback_path: Optional[str]
    subset: Optional[str]
    split: str
    text_field: str
    meta_field: Optional[str]
    source_subfield: Optional[str]
    total_tokens: int
    description: str

    @classmethod
    def from_registry(cls, name: str) -> "DatasetConfig":
        entry = DATASET_REGISTRY[name]
        return cls(name=name, **entry)


@dataclass
class AuditConfig:
    """Top-level configuration for an audit run."""
    datasets: list[str] = field(default_factory=lambda: ["pile"])
    sample_size: int = 100_000
    seed: int = 42
    encoding: str = "cl100k_base"
    calibration_model: str = "claude-haiku-4-5-20251001"
    calibration_sample: int = 600
    output_dir: Path = RESULTS_DIR
    figures_dir: Path = FIGURES_DIR

    def dataset_configs(self) -> list[DatasetConfig]:
        return [DatasetConfig.from_registry(name) for name in self.datasets]
