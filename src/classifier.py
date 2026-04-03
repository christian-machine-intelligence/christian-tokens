"""Keyword-based document classifier for religious traditions.

Uses a three-tier keyword system for maximum defensibility:
- ANCHOR: Words that almost never appear outside explicitly religious content.
  At least one anchor (or verse reference) must be present. Count as 3 hits.
- STRONG: Words that strongly suggest religious content but have occasional
  secular use. Count as 1 hit. Qualify a group toward min_groups.
- SUPPORTING: Words common in religious content but also in secular contexts.
  Count as 1 hit but do NOT qualify a group toward min_groups alone.

Negative filters cancel specific keyword hits when secular context patterns
are detected (e.g., "Martin Luther King" cancels the "Martin Luther" hit).
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import LEXICON_DIR, TRADITIONS

# Keyword tiers
ANCHOR = "anchor"
STRONG = "strong"
SUPPORTING = "supporting"

TIER_WEIGHT = {
    ANCHOR: 3,
    STRONG: 1,
    SUPPORTING: 1,
}


@dataclass
class ClassificationResult:
    """Result of classifying a single document."""
    category: str  # tradition name or "general"
    hits: int  # total weighted keyword matches
    anchor_hits: int = 0  # count of anchor keyword matches
    groups_hit: int = 0  # groups with at least one anchor or strong hit
    hit_details: dict[str, int] = field(default_factory=dict)  # group -> count
    verse_hits: int = 0  # regex verse reference matches


@dataclass
class CompiledKeyword:
    """A compiled keyword pattern with metadata."""
    pattern: re.Pattern
    group: str
    tier: str  # "anchor", "strong", or "supporting"


@dataclass
class NegativeFilter:
    """A negative pattern that cancels specific keyword contributions."""
    pattern: re.Pattern
    cancels: list[str]  # keyword strings whose hits should be removed


@dataclass
class LexiconEntry:
    """A loaded lexicon for one tradition."""
    name: str
    min_hits: int
    min_groups: int
    require_anchor: bool
    compiled_keywords: list[CompiledKeyword] = field(default_factory=list)
    verse_patterns: list[re.Pattern] = field(default_factory=list)
    negative_filters: list[NegativeFilter] = field(default_factory=list)


class KeywordClassifier:
    """Classifies documents by religious tradition using keyword lexicons.

    Traditions are checked in priority order (TRADITIONS constant).
    A document must meet ALL of:
      1. total_weighted_hits >= min_hits
      2. groups with anchor/strong hits >= min_groups
      3. At least 1 anchor keyword or verse reference present
    """

    def __init__(self, lexicon_dir: Path = LEXICON_DIR):
        self.lexicons: list[LexiconEntry] = []
        for tradition in TRADITIONS:
            path = lexicon_dir / f"{tradition}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing lexicon: {path}")
            self.lexicons.append(self._load_lexicon(tradition, path))

    def _load_lexicon(self, name: str, path: Path) -> LexiconEntry:
        """Load and compile a lexicon from JSON."""
        with open(path) as f:
            data = json.load(f)

        entry = LexiconEntry(
            name=name,
            min_hits=data.get("min_hits", 8),
            min_groups=data.get("min_groups", 3),
            require_anchor=data.get("require_anchor", True),
        )

        # Compile keywords — support both old flat format and new tiered format
        keywords = data.get("keywords", {})
        for group, group_data in keywords.items():
            if isinstance(group_data, dict):
                # New tiered format: {"anchor": [...], "strong": [...], "supporting": [...]}
                for tier in [ANCHOR, STRONG, SUPPORTING]:
                    for kw in group_data.get(tier, []):
                        pattern = re.compile(
                            r"\b" + re.escape(kw) + r"\b",
                            re.IGNORECASE,
                        )
                        entry.compiled_keywords.append(
                            CompiledKeyword(pattern=pattern, group=group, tier=tier)
                        )
            elif isinstance(group_data, list):
                # Old flat format — treat all as strong for backward compat
                for kw in group_data:
                    pattern = re.compile(
                        r"\b" + re.escape(kw) + r"\b",
                        re.IGNORECASE,
                    )
                    entry.compiled_keywords.append(
                        CompiledKeyword(pattern=pattern, group=group, tier=STRONG)
                    )

        # Compile verse-reference patterns
        for pat_str in data.get("verse_patterns", []):
            try:
                entry.verse_patterns.append(re.compile(pat_str, re.IGNORECASE))
            except re.error:
                pass

        # Compile negative filters
        for nf in data.get("negative_filters", []):
            try:
                pattern = re.compile(nf["pattern"], re.IGNORECASE)
                entry.negative_filters.append(
                    NegativeFilter(pattern=pattern, cancels=nf.get("cancels", []))
                )
            except (re.error, KeyError):
                pass

        return entry

    def _score_lexicon(self, text: str, lexicon: LexiconEntry) -> ClassificationResult:
        """Score a document against a single lexicon."""
        # Phase 1: Find all keyword hits
        # Track per-keyword hits so negative filters can cancel them
        keyword_hits: dict[str, dict] = {}  # kw_text -> {group, tier, count}

        for ckw in lexicon.compiled_keywords:
            matches = len(ckw.pattern.findall(text))
            if matches > 0:
                kw_text = ckw.pattern.pattern  # use regex pattern as key
                keyword_hits[kw_text] = {
                    "group": ckw.group,
                    "tier": ckw.tier,
                    "count": matches,
                }

        # Phase 2: Apply negative filters — cancel specific keyword hits
        for nf in lexicon.negative_filters:
            if nf.pattern.search(text):
                # Cancel hits from keywords in the cancels list
                cancelled_patterns = set()
                for cancel_kw in nf.cancels:
                    cancel_pattern = r"\b" + re.escape(cancel_kw) + r"\b"
                    cancelled_patterns.add(cancel_pattern)
                # Remove matching keyword hits
                keyword_hits = {
                    k: v for k, v in keyword_hits.items()
                    if k not in cancelled_patterns
                }

        # Phase 3: Aggregate scores
        hit_details: dict[str, int] = {}
        total_hits = 0
        anchor_count = 0
        # Track which groups have anchor or strong hits (for min_groups)
        qualified_groups: set[str] = set()

        for kw_text, info in keyword_hits.items():
            group = info["group"]
            tier = info["tier"]
            count = info["count"]
            weight = TIER_WEIGHT[tier]
            weighted = count * weight

            hit_details[group] = hit_details.get(group, 0) + weighted
            total_hits += weighted

            if tier == ANCHOR:
                anchor_count += count
                qualified_groups.add(group)
            elif tier == STRONG:
                qualified_groups.add(group)
            # SUPPORTING does not qualify the group

        # Phase 4: Verse-reference hits (count as 3 each, anchor-equivalent)
        verse_hits = 0
        for pat in lexicon.verse_patterns:
            verse_hits += len(pat.findall(text))
        total_hits += verse_hits * 3
        if verse_hits > 0:
            anchor_count += verse_hits
            qualified_groups.add("_verse_references")

        return ClassificationResult(
            category=lexicon.name,
            hits=total_hits,
            anchor_hits=anchor_count,
            groups_hit=len(qualified_groups),
            hit_details=hit_details,
            verse_hits=verse_hits,
        )

    def classify(self, text: str) -> ClassificationResult:
        """Classify a document into a religious tradition or 'general'.

        Checks traditions in priority order. First to meet ALL thresholds wins.
        """
        for lexicon in self.lexicons:
            result = self._score_lexicon(text, lexicon)

            meets_hits = result.hits >= lexicon.min_hits
            meets_groups = result.groups_hit >= lexicon.min_groups
            meets_anchor = (
                result.anchor_hits >= 1 if lexicon.require_anchor else True
            )

            if meets_hits and meets_groups and meets_anchor:
                return result

        return ClassificationResult(category="general", hits=0)

    def classify_all(self, text: str) -> dict[str, ClassificationResult]:
        """Classify against ALL traditions (for calibration/analysis)."""
        return {
            lexicon.name: self._score_lexicon(text, lexicon)
            for lexicon in self.lexicons
        }
