"""LLM calibration phase: gold-label a subsample to measure keyword classifier accuracy."""

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tabulate import tabulate

from .config import AuditConfig, TRADITIONS, RESULTS_DIR
from .sampler import AuditResult
from .tokenizer import count_tokens


CLASSIFICATION_PROMPT = """You are a document classifier. Classify the following text excerpt into exactly one of these categories:

1. **christian** — Content primarily about Christianity: Bible/scripture, theology, sermons, catechisms, apologetics, church history, Christian devotion
2. **islamic** — Content primarily about Islam: Quran, hadith, Islamic law/jurisprudence, Sufi literature, Islamic theology
3. **jewish** — Content primarily about Judaism: Torah, Talmud, rabbinical commentary, Kabbalah, Jewish philosophy
4. **hindu** — Content primarily about Hinduism: Vedas, Upanishads, Bhagavad Gita, Hindu philosophy, dharma (Hindu context)
5. **buddhist** — Content primarily about Buddhism: sutras, Pali Canon, Buddhist dharma, meditation traditions, Buddhist philosophy
6. **general** — Does not primarily belong to any of the above religious traditions

Respond with ONLY the category name (one word, lowercase). No explanation.

Text excerpt:
---
{text}
---

Category:"""


@dataclass
class CalibrationResult:
    """Results of the LLM calibration phase."""
    total_samples: int = 0
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    # confusion_matrix[keyword_label][llm_label] = count
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    # per_category[category] = {precision, recall, f1, support}

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "confusion_matrix": self.confusion_matrix,
            "per_category": self.per_category,
        }


def _select_calibration_sample(
    audit_result: AuditResult,
    sample_size: int = 600,
    seed: int = 42,
) -> list[dict]:
    """Select a stratified subsample for calibration.

    Aims for ~100 per religious tradition found, plus random negatives.
    Returns list of {text, keyword_label, dataset, index}.
    """
    # This is a placeholder — in practice we'd need to re-stream and
    # save classified documents during the audit phase.
    # For now, return instructions for the user.
    print(
        "Note: Calibration requires document texts to be saved during audit.\n"
        "Re-run the audit with --save-texts to enable calibration.\n"
        "This is a planned feature for Phase 2."
    )
    return []


def _call_llm(text: str, model: str) -> str:
    """Classify a single document using Claude."""
    import anthropic

    client = anthropic.Anthropic()

    # Truncate to ~500 tokens worth of text (roughly 2000 chars)
    truncated = text[:2000]

    message = client.messages.create(
        model=model,
        max_tokens=10,
        temperature=0,
        messages=[
            {"role": "user", "content": CLASSIFICATION_PROMPT.format(text=truncated)}
        ],
    )

    response = message.content[0].text.strip().lower()

    # Validate response
    valid = set(TRADITIONS) | {"general"}
    if response not in valid:
        return "general"
    return response


def _compute_metrics(confusion: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1 from confusion matrix."""
    all_categories = set(TRADITIONS) | {"general"}
    metrics = {}

    for cat in all_categories:
        tp = confusion.get(cat, {}).get(cat, 0)

        # Precision: of all keyword predicted as cat, how many does LLM agree?
        predicted_as_cat = sum(confusion.get(cat, {}).values())
        precision = tp / predicted_as_cat if predicted_as_cat > 0 else 0

        # Recall: of all LLM labeled as cat, how many did keyword catch?
        llm_labeled_cat = sum(
            row.get(cat, 0) for row in confusion.values()
        )
        recall = tp / llm_labeled_cat if llm_labeled_cat > 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics[cat] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": predicted_as_cat,
        }

    return metrics


def run_calibration(audit_result: AuditResult, config: AuditConfig) -> None:
    """Run the LLM calibration phase."""
    print("LLM Calibration Phase")
    print("=" * 60)
    print(f"Model: {config.calibration_model}")
    print(f"Target sample: {config.calibration_sample} documents\n")

    samples = _select_calibration_sample(
        audit_result,
        sample_size=config.calibration_sample,
        seed=config.seed,
    )

    if not samples:
        print("No samples available for calibration. Skipping.\n")
        print("To enable calibration, the audit phase needs to save")
        print("document texts. This will be implemented in a future version.")
        return

    # Classify each sample with LLM
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        keyword_label = sample["keyword_label"]
        llm_label = _call_llm(sample["text"], config.calibration_model)
        confusion[keyword_label][llm_label] += 1

    # Compute metrics
    metrics = _compute_metrics(dict(confusion))

    # Print report
    print("\nCalibration Report")
    print("=" * 60)

    rows = []
    for cat in TRADITIONS + ["general"]:
        m = metrics.get(cat, {})
        rows.append([
            cat.title(),
            f"{m.get('precision', 0):.3f}",
            f"{m.get('recall', 0):.3f}",
            f"{m.get('f1', 0):.3f}",
            m.get("support", 0),
        ])

    headers = ["Category", "Precision", "Recall", "F1", "Support"]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Save calibration results
    result = CalibrationResult(
        total_samples=len(samples),
        confusion_matrix=dict(confusion),
        per_category=metrics,
    )

    path = config.output_dir / f"calibration_{audit_result.timestamp}.json"
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nCalibration results saved to: {path}")
