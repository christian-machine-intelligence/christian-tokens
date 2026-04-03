"""HuggingFace dataset streaming wrappers."""

from dataclasses import dataclass
from typing import Iterator, Optional

from datasets import load_dataset

from .config import DatasetConfig


@dataclass
class Document:
    """A normalized document from any dataset."""
    text: str
    source: Optional[str]  # e.g., "Pile-CC", "Gutenberg" for The Pile
    dataset_name: str
    index: int


def _extract_source(row: dict, config: DatasetConfig) -> Optional[str]:
    """Extract the source subset label from a dataset row."""
    if config.meta_field is None or config.source_subfield is None:
        return None
    meta = row.get(config.meta_field)
    if meta is None:
        return None
    if isinstance(meta, dict):
        return meta.get(config.source_subfield)
    return None


def stream_dataset(
    config: DatasetConfig,
    sample_size: int,
    seed: int = 42,
) -> Iterator[Document]:
    """Stream a random sample of documents from a HuggingFace dataset.

    Tries the primary path first; falls back if gated/unavailable.
    """
    ds = None
    for path in [config.hf_path, config.fallback_path]:
        if path is None:
            continue
        try:
            ds = load_dataset(
                path,
                name=config.subset,
                split=config.split,
                streaming=True,
            )
            break
        except Exception as e:
            print(f"  Warning: could not load {path}: {e}")
            continue

    if ds is None:
        raise RuntimeError(
            f"Could not load dataset {config.name} from any source. "
            f"Tried: {config.hf_path}, {config.fallback_path}"
        )

    # Shuffle with seed for reproducibility, then take sample
    ds = ds.shuffle(seed=seed).take(sample_size)

    for i, row in enumerate(ds):
        text = row.get(config.text_field, "")
        if not text:
            continue
        yield Document(
            text=text,
            source=_extract_source(row, config),
            dataset_name=config.name,
            index=i,
        )
