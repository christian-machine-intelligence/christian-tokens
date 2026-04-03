"""Visualization: matplotlib/seaborn figures for the audit results."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from .config import TRADITIONS
from .statistics import wilson_ci, extrapolate, format_tokens


# Color palette for traditions
TRADITION_COLORS = {
    "christian": "#2563EB",   # blue
    "islamic": "#059669",     # green
    "jewish": "#7C3AED",      # purple
    "hindu": "#DC2626",       # red
    "buddhist": "#D97706",    # amber
    "general": "#9CA3AF",     # gray
}


def fig_tradition_proportions(data: dict, output_dir: Path) -> None:
    """Stacked bar chart: religious content proportions across datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))

    dataset_names = []
    tradition_pcts = {t: [] for t in TRADITIONS}

    for ds in data.get("dataset_results", []):
        dataset_names.append(ds["dataset_name"].title())
        sample_tokens = ds["sample_tokens"]
        for t in TRADITIONS:
            tally = ds["tallies"].get(t, {"token_count": 0})
            pct = tally["token_count"] / sample_tokens * 100 if sample_tokens > 0 else 0
            tradition_pcts[t].append(pct)

    bottom = [0] * len(dataset_names)
    for t in TRADITIONS:
        bars = ax.bar(
            dataset_names,
            tradition_pcts[t],
            bottom=bottom,
            label=t.title(),
            color=TRADITION_COLORS[t],
        )
        bottom = [b + v for b, v in zip(bottom, tradition_pcts[t])]

    ax.set_ylabel("% of Corpus Tokens")
    ax.set_title("Religious Content in LLM Training Datasets")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())

    plt.tight_layout()
    path = output_dir / "tradition_proportions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_extrapolated_tokens(data: dict, output_dir: Path) -> None:
    """Log-scale bar chart: extrapolated religious token counts."""
    ds = data["dataset_results"][0] if data.get("dataset_results") else None
    if ds is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = []
    values = []
    colors = []

    sample_tokens = ds["sample_tokens"]
    total_tokens = ds["total_tokens_known"]

    for t in TRADITIONS:
        tally = ds["tallies"].get(t, {"token_count": 0})
        if tally["token_count"] == 0:
            continue
        prop = wilson_ci(tally["token_count"], sample_tokens)
        ext = extrapolate(prop, total_tokens)
        if ext.est_tokens > 0:
            labels.append(t.title())
            values.append(ext.est_tokens)
            colors.append(TRADITION_COLORS[t])

    if not values:
        plt.close(fig)
        return

    bars = ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("Estimated Tokens (log scale)")
    ax.set_title(f"Estimated Religious Content in Full Corpus\n({ds['description']})")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
            format_tokens(val),
            va="center", fontsize=9,
        )

    plt.tight_layout()
    path = output_dir / "extrapolated_tokens.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_source_breakdown(data: dict, output_dir: Path) -> None:
    """Heatmap: religious content by source subset (for datasets with metadata)."""
    for ds in data.get("dataset_results", []):
        breakdown = ds.get("source_breakdown", {})
        if not breakdown:
            continue

        sources = sorted(breakdown.keys())
        if len(sources) < 2:
            continue

        matrix = []
        source_labels = []
        for src in sources:
            cats = breakdown[src]
            total = sum(c["token_count"] for c in cats.values())
            if total == 0:
                continue
            row = []
            for t in TRADITIONS:
                pct = cats.get(t, {"token_count": 0})["token_count"] / total * 100
                row.append(pct)
            matrix.append(row)
            source_labels.append(src)

        if not matrix:
            continue

        fig, ax = plt.subplots(figsize=(10, max(6, len(source_labels) * 0.4)))
        sns.heatmap(
            matrix,
            xticklabels=[t.title() for t in TRADITIONS],
            yticklabels=source_labels,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            ax=ax,
        )
        ax.set_title(f"Religious Content % by Source: {ds['dataset_name'].title()}")
        ax.set_xlabel("Tradition")
        ax.set_ylabel("Source Subset")

        plt.tight_layout()
        path = output_dir / f"source_breakdown_{ds['dataset_name']}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def fig_forest_plot(data: dict, output_dir: Path) -> None:
    """Forest plot: confidence intervals for each tradition across datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = 0
    y_labels = []
    y_positions = []

    for ds in data.get("dataset_results", []):
        sample_tokens = ds["sample_tokens"]
        for t in TRADITIONS:
            tally = ds["tallies"].get(t, {"token_count": 0})
            prop = wilson_ci(tally["token_count"], sample_tokens)

            ax.errorbar(
                prop.proportion * 100,
                y_pos,
                xerr=[[
                    (prop.proportion - prop.ci_lower) * 100
                ], [
                    (prop.ci_upper - prop.proportion) * 100
                ]],
                fmt="o",
                color=TRADITION_COLORS[t],
                capsize=4,
                markersize=6,
            )
            y_labels.append(f"{t.title()} ({ds['dataset_name']})")
            y_positions.append(y_pos)
            y_pos += 1
        y_pos += 0.5  # gap between datasets

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("% of Corpus Tokens")
    ax.set_title("Religious Content Proportions with 95% Confidence Intervals")
    ax.invert_yaxis()

    plt.tight_layout()
    path = output_dir / "forest_plot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_all_figures(data: dict, output_dir: Path) -> None:
    """Generate all figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating figures in {output_dir}/...")

    fig_tradition_proportions(data, output_dir)
    fig_extrapolated_tokens(data, output_dir)
    fig_source_breakdown(data, output_dir)
    fig_forest_plot(data, output_dir)

    print("Done.\n")
