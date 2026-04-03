"""Main audit pipeline: stream, classify, and count tokens."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .classifier import KeywordClassifier
from .config import AuditConfig, DatasetConfig, TRADITIONS
from .datasets import stream_dataset
from .tokenizer import count_tokens


@dataclass
class CategoryTally:
    """Accumulated counts for one category."""
    doc_count: int = 0
    token_count: int = 0


@dataclass
class DatasetResult:
    """Results for a single dataset."""
    dataset_name: str
    description: str
    total_tokens_known: int  # published total for the full dataset
    sample_size_target: int
    docs_processed: int = 0
    sample_tokens: int = 0  # total tokens in our sample
    tallies: dict[str, CategoryTally] = field(default_factory=dict)
    source_breakdown: dict[str, dict[str, CategoryTally]] = field(
        default_factory=dict
    )  # source_subset -> category -> tally

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "description": self.description,
            "total_tokens_known": self.total_tokens_known,
            "sample_size_target": self.sample_size_target,
            "docs_processed": self.docs_processed,
            "sample_tokens": self.sample_tokens,
            "tallies": {
                cat: {"doc_count": t.doc_count, "token_count": t.token_count}
                for cat, t in self.tallies.items()
            },
            "source_breakdown": {
                src: {
                    cat: {"doc_count": t.doc_count, "token_count": t.token_count}
                    for cat, t in cats.items()
                }
                for src, cats in self.source_breakdown.items()
            },
        }


@dataclass
class AuditResult:
    """Full audit results across all datasets."""
    timestamp: str
    config: dict
    dataset_results: list[DatasetResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "config": self.config,
            "dataset_results": [r.to_dict() for r in self.dataset_results],
        }

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"audit_{self.timestamp}.json"
        path = output_dir / filename
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def _init_tallies() -> dict[str, CategoryTally]:
    """Initialize tally dict with all categories."""
    tallies = {t: CategoryTally() for t in TRADITIONS}
    tallies["general"] = CategoryTally()
    return tallies


def audit_dataset(
    ds_config: DatasetConfig,
    classifier: KeywordClassifier,
    sample_size: int,
    seed: int,
    encoding: str,
    save_interval: int = 10_000,
    output_dir: Optional[Path] = None,
) -> DatasetResult:
    """Run the audit pipeline on a single dataset."""
    result = DatasetResult(
        dataset_name=ds_config.name,
        description=ds_config.description,
        total_tokens_known=ds_config.total_tokens,
        sample_size_target=sample_size,
        tallies=_init_tallies(),
    )

    print(f"\n{'='*60}")
    print(f"Auditing: {ds_config.description}")
    print(f"Sample size: {sample_size:,} documents")
    print(f"{'='*60}\n")

    stream = stream_dataset(ds_config, sample_size=sample_size, seed=seed)
    pbar = tqdm(stream, total=sample_size, desc=ds_config.name)

    for doc in pbar:
        # Classify
        classification = classifier.classify(doc.text)
        category = classification.category

        # Count tokens
        tokens = count_tokens(doc.text, encoding)

        # Update tallies
        result.tallies[category].doc_count += 1
        result.tallies[category].token_count += tokens
        result.sample_tokens += tokens
        result.docs_processed += 1

        # Source-level breakdown (e.g., Pile subsets)
        if doc.source:
            if doc.source not in result.source_breakdown:
                result.source_breakdown[doc.source] = _init_tallies()
            result.source_breakdown[doc.source][category].doc_count += 1
            result.source_breakdown[doc.source][category].token_count += tokens

        # Progress update
        if result.docs_processed % 1000 == 0:
            christian = result.tallies["christian"]
            pct = (
                christian.doc_count / result.docs_processed * 100
                if result.docs_processed > 0
                else 0
            )
            pbar.set_postfix(
                christian=f"{pct:.1f}%",
                tokens=f"{result.sample_tokens:,}",
            )

    pbar.close()
    return result


def run_audit(config: AuditConfig) -> AuditResult:
    """Run the full audit across all configured datasets."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    classifier = KeywordClassifier()

    audit_result = AuditResult(
        timestamp=timestamp,
        config={
            "datasets": config.datasets,
            "sample_size": config.sample_size,
            "seed": config.seed,
            "encoding": config.encoding,
        },
    )

    for ds_config in config.dataset_configs():
        ds_result = audit_dataset(
            ds_config=ds_config,
            classifier=classifier,
            sample_size=config.sample_size,
            seed=config.seed,
            encoding=config.encoding,
            output_dir=config.output_dir,
        )
        audit_result.dataset_results.append(ds_result)

    # Save results
    path = audit_result.save(config.output_dir)
    print(f"\nResults saved to: {path}")

    return audit_result
