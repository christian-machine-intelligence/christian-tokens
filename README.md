# christian-tokens

**Quantifying Religious Content in LLM Pretraining Data**

A corpus analysis study measuring the proportion of explicitly religious content — with a focus on Christianity — in open-source LLM pretraining datasets. Produced by the [Institute for Christian Machine Intelligence](https://github.com/christian-machine-intelligence).

## Key Finding

Approximately **67 billion tokens (~8.1%)** of The Pile — a foundational open-source pretraining corpus — consist of explicitly Christian content: scripture, theology, sermons, catechisms, apologetics, and church history. This makes Christianity by far the largest religious tradition represented in LLM training data, exceeding the next largest (Buddhism, at ~3.5B tokens) by nearly 20x.

These numbers use the strictest possible classification methodology and measure only **Tier 1 (explicit) content** — documents where Christianity is the primary subject matter. They do not capture Tier 2 (Christian-inflected moral reasoning in secular contexts) or Tier 3 (Western literature steeped in biblical allusion), both of which are far larger. Our estimates are a conservative lower bound.

### What does 67 billion tokens mean?

- **15x the entire English Wikipedia.** All of Wikipedia, on every topic, is ~4.4B tokens.
- **~890,000 books** of theology, sermons, scripture, commentary, and church history.
- **1.7x the entire GPT-2 training set.** GPT-2 was trained on 40B tokens total, across all subjects.
- **22% of GPT-3's training volume.** One in five tokens GPT-3 learned from is comparable in scale to the Christian moral corpus.
- **~1,675x the entire AI safety research literature.** An estimated ~5,000 published AI safety papers at ~8K tokens each yields roughly 40M tokens. Christian moral reasoning in pretraining exceeds this by three orders of magnitude.
- **383 years of continuous reading** at 250 words per minute, 24/7, without stopping.

## Results

### Religious Content in The Pile (100,000-document sample, 154.4M tokens)

| Tradition | Docs | Sample Tokens | % of Corpus | Est. Full (825B) | 95% CI |
|-----------|-----:|:--------------|:-----------:|:----------------:|:------:|
| Christian | 287 | 12,549,126 | 8.125% | ~67.0B | [67.0B, 67.1B] |
| Islamic | 29 | 389,155 | 0.252% | ~2.1B | [2.1B, 2.1B] |
| Buddhist | 14 | 652,227 | 0.422% | ~3.5B | [3.5B, 3.5B] |
| Hindu | 20 | 286,161 | 0.185% | ~1.5B | [1.5B, 1.5B] |
| Jewish | 25 | 135,271 | 0.088% | ~723M | [719M, 726M] |
| **All Religious** | **375** | **14,011,940** | **9.072%** | **~74.8B** | **[74.8B, 74.9B]** |

### Source Breakdown (The Pile subsets)

| Source | Christian % | All Religious % | Notes |
|--------|:-----------:|:---------------:|-------|
| Gutenberg (PG-19) | 69.2% | 69.2% | Public domain books; heavy Bible and theology |
| Books3 | 47.6% | 52.8% | Broad book corpus; nearly half Christian |
| YouTube Subtitles | 4.2% | 4.2% | Sermons, religious media |
| Pile-CC | 2.9% | 4.0% | Common Crawl; web-scale signal |
| PhilPapers | 2.7% | 2.7% | Philosophy of religion |
| Wikipedia (en) | 2.6% | 3.9% | Church history, theology articles |
| OpenWebText2 | 1.1% | 1.3% | Reddit-sourced web content |
| ArXiv | 0.0% | 0.0% | Scientific papers (no signal, as expected) |
| GitHub | 0.0% | 0.0% | Source code (no signal, as expected) |
| PubMed | 0.0% | 0.0% | Biomedical literature (no signal, as expected) |

Christian content appears across multiple independent sources — it is not an artifact of any single subset.

## Methodology

### Three-Tier Keyword Classification

Documents are classified into five religious traditions using keyword lexicons with a strict three-tier system designed to minimize false positives:

- **Anchor keywords** (weight: 3x) — Words that almost never appear outside explicitly religious content (e.g., "Christology", "transubstantiation", "Nicene Creed", "Jesus Christ", "Holy Spirit"). **At least one anchor keyword must be present** for a document to be classified.
- **Strong keywords** (weight: 1x) — Words that strongly suggest religious content but have occasional secular use (e.g., "Bible", "theology", "liturgy"). Qualify a keyword group toward the breadth threshold.
- **Supporting keywords** (weight: 1x) — Words common in both religious and secular contexts (e.g., "Baptist", "Evangelical", "Messiah"). Contribute to hit count but do **not** qualify a group on their own.

A document must meet **all three** requirements to be classified:
1. **Total weighted hits >= 8**
2. **Hits across >= 3 distinct keyword groups** (with anchor or strong presence)
3. **At least 1 anchor keyword or verse reference present**

**Negative filters** cancel specific keyword hits when secular context is detected (e.g., "Martin Luther King" cancels the "Martin Luther" hit; "Sega Genesis" cancels "Genesis").

All five tradition lexicons (Christian, Islamic, Jewish, Hindu, Buddhist) use identical threshold requirements and the same strictness standard.

### Traditions

1. **Christian** — Bible/scripture, theology, sermons, catechisms, apologetics, church history
2. **Islamic** — Quran, hadith, Islamic jurisprudence, Sufi literature, Islamic theology
3. **Jewish** — Torah, Talmud, rabbinical commentary, Kabbalah, Jewish philosophy
4. **Hindu** — Vedas, Upanishads, Bhagavad Gita, Hindu philosophy, dharma
5. **Buddhist** — Sutras, Pali Canon, Buddhist dharma, meditation traditions

### Statistical Design

- **Sampling**: 100,000 documents streamed from HuggingFace with reproducible shuffling (seed=42)
- **Token counting**: tiktoken with cl100k_base encoding (GPT-4 tokenizer)
- **Confidence intervals**: Wilson score intervals (better than Wald for small proportions)
- **Extrapolation**: Sample proportions projected to full dataset sizes

### Datasets

| Dataset | Size | Source | Status |
|---------|------|--------|--------|
| The Pile | ~825B tokens | EleutherAI | **Complete** |
| Dolma v1.7 | ~3T tokens | AI2 (OLMo) | Planned |
| FineWeb | ~15T tokens | HuggingFace | Planned |

## Usage

```bash
# Set up
cd christian-tokens
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Quick smoke test (1000 docs, Pile only)
python -m src --quick

# Full audit on The Pile (100K docs, ~2 hours)
python -m src

# Audit multiple datasets
python -m src --dataset pile dolma fineweb

# Custom sample size
python -m src --sample-size 50000

# Regenerate tables from saved results
python -m src --analyze results/audit_TIMESTAMP.json

# Regenerate figures from saved results
python -m src --visualize results/audit_TIMESTAMP.json
```

## Project Structure

```
christian-tokens/
├── README.md
├── requirements.txt
├── data/
│   └── lexicons/              # Keyword lists per tradition (JSON)
│       ├── christian.json     # 7 keyword groups, 97 anchors
│       ├── islamic.json
│       ├── jewish.json
│       ├── hindu.json
│       └── buddhist.json
├── src/
│   ├── __main__.py            # CLI entry point
│   ├── config.py              # Dataset registry, paths, dataclasses
│   ├── datasets.py            # HuggingFace streaming wrappers
│   ├── classifier.py          # Three-tier keyword classifier
│   ├── tokenizer.py           # tiktoken token counting
│   ├── sampler.py             # Main audit pipeline
│   ├── calibrate.py           # LLM gold-label calibration (optional)
│   ├── statistics.py          # Wilson CIs, z-tests, extrapolation
│   ├── analysis.py            # Tabulate-based reporting
│   └── visualize.py           # Matplotlib/seaborn figures
├── results/                   # JSON audit results
└── figures/                   # Generated figures (PNG)
```

## Dependencies

- `datasets` — HuggingFace dataset streaming
- `tiktoken` — Token counting (cl100k_base)
- `scipy` — Confidence intervals and statistical tests
- `tabulate` — Table formatting
- `tqdm` — Progress bars
- `matplotlib` / `seaborn` — Figures
- `anthropic` — LLM calibration phase (optional)
- `zstandard` — Decompression for streaming datasets
