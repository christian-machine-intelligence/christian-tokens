"""Analysis and reporting using tabulate."""

from tabulate import tabulate

from .config import TRADITIONS
from .statistics import (
    wilson_ci,
    extrapolate,
    format_tokens,
)


def print_tradition_table(ds_result: dict) -> None:
    """Print Table 1: Religious content breakdown for one dataset."""
    sample_tokens = ds_result["sample_tokens"]
    total_tokens = ds_result["total_tokens_known"]
    docs_processed = ds_result["docs_processed"]
    tallies = ds_result["tallies"]

    print(f"\n{'='*80}")
    print(f"  {ds_result['description']}")
    print(f"  Sample: {docs_processed:,} docs, {format_tokens(sample_tokens)} tokens")
    print(f"{'='*80}\n")

    rows = []
    all_religious_tokens = 0
    all_religious_docs = 0

    for tradition in TRADITIONS:
        tally = tallies.get(tradition, {"doc_count": 0, "token_count": 0})
        doc_count = tally["doc_count"]
        token_count = tally["token_count"]

        if tradition != "general":
            all_religious_tokens += token_count
            all_religious_docs += doc_count

        # Proportion based on tokens
        prop = wilson_ci(token_count, sample_tokens) if sample_tokens > 0 else None
        ext = extrapolate(prop, total_tokens) if prop else None

        rows.append([
            tradition.title(),
            f"{doc_count:,}",
            f"{token_count:,}",
            f"{prop.proportion * 100:.3f}%" if prop else "—",
            f"[{prop.ci_lower * 100:.3f}%, {prop.ci_upper * 100:.3f}%]" if prop else "—",
            format_tokens(ext.est_tokens) if ext else "—",
            f"[{format_tokens(ext.ci_lower)}, {format_tokens(ext.ci_upper)}]" if ext else "—",
        ])

    # Add combined religious row
    prop_all = wilson_ci(all_religious_tokens, sample_tokens) if sample_tokens > 0 else None
    ext_all = extrapolate(prop_all, total_tokens) if prop_all else None
    rows.append(["—"] * 7)
    rows.append([
        "ALL RELIGIOUS",
        f"{all_religious_docs:,}",
        f"{all_religious_tokens:,}",
        f"{prop_all.proportion * 100:.3f}%" if prop_all else "—",
        f"[{prop_all.ci_lower * 100:.3f}%, {prop_all.ci_upper * 100:.3f}%]" if prop_all else "—",
        format_tokens(ext_all.est_tokens) if ext_all else "—",
        f"[{format_tokens(ext_all.ci_lower)}, {format_tokens(ext_all.ci_upper)}]" if ext_all else "—",
    ])

    # General (non-religious)
    general = tallies.get("general", {"doc_count": 0, "token_count": 0})
    rows.append([
        "General",
        f"{general['doc_count']:,}",
        f"{general['token_count']:,}",
        f"{general['token_count'] / sample_tokens * 100:.3f}%" if sample_tokens > 0 else "—",
        "", "", "",
    ])

    headers = [
        "Tradition", "Docs", "Sample Tokens",
        "% of Corpus", "95% CI",
        "Est. Full", "Est. CI",
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_source_breakdown(ds_result: dict) -> None:
    """Print Table 2: Per-source breakdown (e.g., Pile subsets)."""
    source_breakdown = ds_result.get("source_breakdown", {})
    if not source_breakdown:
        return

    print(f"\n{'='*80}")
    print(f"  SOURCE BREAKDOWN: {ds_result['description']}")
    print(f"{'='*80}\n")

    rows = []
    for source in sorted(source_breakdown.keys()):
        cats = source_breakdown[source]
        total_docs = sum(c["doc_count"] for c in cats.values())
        total_tokens = sum(c["token_count"] for c in cats.values())
        christian = cats.get("christian", {"doc_count": 0, "token_count": 0})

        if total_docs == 0:
            continue

        christian_pct = christian["token_count"] / total_tokens * 100 if total_tokens > 0 else 0

        # All religious
        religious_tokens = sum(
            c["token_count"] for name, c in cats.items() if name != "general"
        )
        religious_pct = religious_tokens / total_tokens * 100 if total_tokens > 0 else 0

        rows.append([
            source,
            f"{total_docs:,}",
            f"{format_tokens(total_tokens)}",
            f"{christian_pct:.1f}%",
            f"{religious_pct:.1f}%",
        ])

    # Sort by Christian %
    rows.sort(key=lambda r: float(r[3].replace("%", "")), reverse=True)

    headers = ["Source Subset", "Docs", "Tokens", "Christian %", "All Religious %"]
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_full_report(data: dict) -> None:
    """Print the complete analysis report."""
    print("\n" + "=" * 80)
    print("  TRAINING DATA RELIGIOUS CONTENT AUDIT")
    print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
    print("=" * 80)

    for ds_result in data.get("dataset_results", []):
        print_tradition_table(ds_result)
        print_source_breakdown(ds_result)
