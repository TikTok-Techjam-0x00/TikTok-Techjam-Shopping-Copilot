# TikTok TechJam Shopping Copilot

Team 0x00's conversational shopping agent for the TechJam 2026 Track 4 challenge.
It maintains shopping intent and constraints, retrieves catalog products,
reranks them using product evidence, and asks clarification questions while
returning recommendations.

Final repository: [TikTok-Techjam-Shopping-Copilot](https://github.com/TikTok-Techjam-0x00/TikTok-Techjam-Shopping-Copilot).
This release is prepared from the team's `sota-2.2` branch at `d5ac6e9` and
preserves the team development history. Release preparation does not itself
establish new benchmark results.

## Quick start

Use Python 3.10 or later. Run all commands from the repository root, with a
virtual environment activated. The integrated agent requires the packages in
`requirements.txt`; it is not the original standard-library-only weak starter.

```bash
python -m venv .venv
```

Activate `.venv` with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on macOS/Linux, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download `catalog.jsonl.gz` and `SHA256SUMS` from the
[official Participant Kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participantkit).
Verify the archive against the published checksum. Use the local preparation
script to decompress and validate the frozen catalog:

```bash
python scripts/prepare_catalog.py --archive /path/to/catalog.jsonl.gz
```

The script does not download data. Its default archive is `catalog.jsonl.gz`
in the repository root; its output is `data/catalog.jsonl`. It validates the
frozen catalog hash. The full upstream Amazon Reviews dataset is not needed.

Create a directory for local results, then run the official evaluator:

```bash
python -c "from pathlib import Path; Path('artifacts/release-validation').mkdir(parents=True, exist_ok=True)"
python -m evaluator.local_evaluator --output artifacts/release-validation/public200.json
```

This evaluates all 200 public sessions. Add `--limit 4` and use a separate
output filename for a quick smoke run. The output contains overall and
scenario-level metrics plus per-session results. Do not edit evaluator logic
or labels when reporting scores. Generated release-validation output is local
and ignored by Git.

Important: the command above uses the original SOTA 2.2 environment-dependent
behavior described below. For a no-key, local BM25 run regardless of a local
`.env` file, explicitly suppress the key in the evaluation process:

```bash
python -c "import os, runpy; os.environ['DASHSCOPE_API_KEY']=''; runpy.run_module('evaluator.local_evaluator', run_name='__main__')" --output artifacts/release-validation/public200.json
```

The explicit empty value is preserved by the current dotenv loading behavior;
it disables automatic Qwen State resolution and Dense credential setup. This
command does not alter your saved `.env` file.

## Architecture and runtime modes

The evaluator imports `Agent` from `starter/agent.py`. Despite that directory's
historical name, this is the team's integrated agent, not the weak starter.

1. **State:** update intent, hard/soft constraints, rejected values, and the
   active constraint epoch from the observable conversation and user profile.
2. **Retrieval:** intent-routed, field-weighted BM25 supplies candidates;
   late turns explore deeper rank windows.
3. **Reranking:** the production pipeline calls the local
   `EvidenceCoverageReranker.rank_all()` through the existing reranker wrapper.
   The Qwen reranking experiment is not the normal production ranking path.
4. **Dialogue:** choose a high-information clarification attribute and return
   recommendations in the same response. Turns 1 and 2 return one product
   each; later turns return up to `top_k`. Exact recommendations are not
   repeated within a constraint epoch; an intent override resets that memory.

The original SOTA 2.2 automatic configuration is retained:

| Configuration | Retrieval behavior | State behavior |
|---|---|---|
| No `DASHSCOPE_API_KEY` in process or `.env` | Local SOTA 2.1 BM25 path | Local rule-based state updates |
| Key configured, but no valid Dense cache/endpoint configuration | BM25 fallback | Qwen semantic resolution may run for ambiguous updates |
| Key, endpoint, and compatible `dense_needs_v1` cache configured | BM25 plus a bounded Dense residual on turn 8 | Qwen semantic resolution may run for ambiguous updates |

Therefore the unmodified default entry point is **not environment-independent**.
Installing a cache or adding a key can change its execution path. A no-key run
requires neither embedding vectors nor live model access. Dependencies must
still be installed before offline execution.

### Optional SOTA 2.2 semantic residual

For the optional Dense path, fetch the Git LFS vector asset and configure your
own endpoint using [`.env.example`](.env.example):

```bash
git lfs pull --include="artifacts/retrieval/dense/text-embedding-v4__dense_needs_v1__d256/embeddings.npy"
```

The required cache directory contains `embeddings.npy`, `manifest.json`, and
`parent_asins.json`. A Git LFS pointer alone is not a usable vector matrix.
The representation is `dense_needs_v1`, using `text-embedding-v4` at 256
dimensions. On turn 8, the focused cohort contains BM25 ranks 201-210 plus up
to 10 candidates independently supported by Dense Top 200 and BM25 ranks
301-1000. Other retrieval turns retain the BM25 route. Missing/incompatible
cache, unavailable credentials, or a Dense provider error falls back to the
original BM25 page.

Copy `.env.example` to an ignored local `.env` only if you intend to use
external services; do not use its placeholder key. Query embeddings and
optional State resolution require network access and can incur costs and
timeout/retry latency. Do not put API keys in Git or reports.

## Validation and recorded results

Run the core unit suite with:

```bash
python -m unittest discover -s . -p "test*.py"
```

Some integration tests construct the default Agent and can read `.env`. To
run them without automatic external services, use:

```bash
python -c "import os, runpy; os.environ['DASHSCOPE_API_KEY']=''; runpy.run_module('unittest', run_name='__main__')" discover -s . -p "test*.py"
```

Frontend tests are discovered separately and need the optional demo dependencies:

```bash
python -m pip install -r frontend/requirements.txt
python -c "import os, runpy; os.environ['DASHSCOPE_API_KEY']=''; runpy.run_module('unittest', run_name='__main__')" discover -s frontend -p "test*.py"
```

The 2026-08-31 release check passed **244 core tests and 9 frontend tests**.
The no-key Public-200 run reproduced TechnicalScore **0.961150**, HitRate@10
**0.995000**, MRR **0.960833**, and MTTC **2.23000** in **50.936 seconds** on
the validation machine (including imports and index construction). See the
[release validation record](results_history/sota_2_2_release/README.md) for
provenance, exact results, environment, and limitations. This verifies the
BM25 fallback, not a live embedding endpoint.

Three generated 800-session generalization sets are provided under
[`data/generalization/v1/`](data/generalization/v1/README.md). They exclude
Public-200 targets, are mutually target-disjoint, and are development-only
evaluation data, not the organizer's private test set. Evaluate one with
`--dataset data/generalization/v1/iid_800.jsonl` and a separate output file.

[`docs/SOTA_2_2_RETRIEVAL.md`](docs/SOTA_2_2_RETRIEVAL.md) records the frozen
algorithm, historical scores, and anti-leakage controls. Its reported
Public-200 result is TechnicalScore `0.961150`, HitRate@10 `0.995000`, MRR
`0.960833`, and MTTC `2.23000`. These are **historical SOTA 2.2 results**, not
a claim that this release preparation has remeasured them. Record the code
commit, dataset, runtime mode, dependencies, elapsed time, and output path for
each new run; do not compare an API-enabled run against a no-key run without
disclosing the difference.

The original organizer weak-starter reference remains in
[`docs/baseline_results.json`](docs/baseline_results.json); it is not the
current agent's result.

## Challenge and Agent interface

The frozen catalog has 50,000 products from Amazon Reviews 2023
`Clothing_Shoes_and_Jewelry`. The public development set has 200 sessions
covering Buying, Browsing, Intent Override, and Boundary behavior. The
organizer's final 800 sessions are private and are not included here.

Each session supplies an anonymized preference profile and customer messages.
The Agent can ask one attribute-specific clarification, recommend up to 10
catalog products, or do both. The official evaluator scores exact
`parent_asin` matches, subject to its intent-override protocol, within at most
10 turns. The Agent does not receive target labels or hidden intent cards.

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Team Pipeline Contracts

The implementation under `src/` uses shared data contracts so that State,
Retrieval, Reranking, Dialogue, and the final Agent exchange the same object
shapes.

### Product and candidate objects

- `Item` represents one product from `data/catalog.jsonl`.
- `Candidate` is a Retrieval result and contains an `Item` plus retrieval
  scores.
- `RankedCandidate` is a Reranking result and contains an `Item` plus reranking
  scores, matched attributes, and hard-constraint violations.
- `Candidates100` and `Candidates10` are the list aliases exchanged between
  Retrieval, Reranking, and Dialogue.

These objects use composition: `Candidate.item` and `RankedCandidate.item`
refer to an `Item`; candidate classes do not inherit from `Item`.

### Standard shopping attributes

`shopping_state.hard_constraint` and `shopping_state.soft_constraint` use the
following common type:

```python
from src.attribute import AttributeMap, AttributeName, normalize_attribute_map

hard_constraint: AttributeMap = normalize_attribute_map({
    "category": ["running shoes"],
    "budget": {"max": 100, "unit": "USD"},
})

soft_constraint: AttributeMap = normalize_attribute_map({
    "color": ["black"],
    "feature": ["lightweight", "comfortable"],
})

no_prefernce: list[AttributeName] = [AttributeName.BRAND]
```

The canonical attribute names exactly match the evaluator's `ask_attribute` values:

```text
category, material, color, size, style, brand,
budget, feature, use_case, other
```

Each entry is an `AttributeValue` with one stable shape:

```python
AttributeValue(
    values=["M"],
    minimum=None,
    maximum=None,
    unit=None,
    details={"waist": ["32 inches"]},
)
```

- `values` stores one or more categorical/text values.
- `minimum` and `maximum` store numeric ranges such as budget.
- `unit` stores the currency or measurement unit.
- `details` stores structured subfields such as product dimensions.

Each catalog `Item` also exposes a lazily computed `item.attributes` mapping.
It derives canonical product attributes from official catalog fields and caches
the result on first use. `item.to_dict()` intentionally continues to return only
the official catalog fields, so evaluator and serialization contracts do not
change.

To export all derived catalog attributes as a separate reproducible JSONL file:

```bash
python extract_attributes.py
```

The default output is `data/catalog_attributes.jsonl`, containing one
`parent_asin` and normalized `attributes` mapping per product. Generated files
are ignored by Git and should not be committed.

Call `normalize_attribute_map()` on rule-based or model-generated extraction
results before saving them in `shopping_state`. It normalizes common catalog
aliases such as `Fabric Type -> material`, `Department -> style`, and
`Occasion -> use_case`. An unknown field is not discarded; it is retained under
`AttributeName.OTHER`:

```python
normalize_attribute_map({"Care Instructions": "hand wash only"})

# Equivalent normalized content:
{
    AttributeName.OTHER: AttributeValue(
        details={"care_instructions": ["hand wash only"]}
    )
}
```

The public contract intentionally contains no extra enum members. Catalog and
details aliases are folded into these ten fields, and
`to_official_ask_attribute()` always returns an evaluator-valid value. The
former richer schema remains available in `src/atrribute_detailed.py` for
comparison or experiments, but production modules import `src.attribute`.

See [`src/reranking/README.md`](src/reranking/README.md) for the full Reranking
input/output contract and runnable examples.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model access, cost, and limitations

The local BM25/evidence-ranking path makes no model API calls. The optional
Dense residual uses the configured embedding provider, and automatic State
resolution uses the configured Qwen-compatible model. Teams supply their own
credentials and pay for any external service usage; no organizer credits are
provided. Network access may be disabled during official evaluation, so the
offline fallback and the selected runtime mode must be disclosed.

The current Agent `usage` payload reports reranker counters only. It does not
aggregate embedding or State-model usage, so a zero reported count must not
be interpreted as proof of zero external-service usage in an API-enabled run.
Report provider usage and measured latency separately if those options are
enabled. Token usage is not part of the core technical score.

Lexical matching can miss paraphrases and ambiguous attributes. The optional
semantic path has only a bounded late-turn effect and depends on a compatible
cache and provider. Public and generated-set scores do not guarantee hidden
test performance. Evaluation datasets and labels must stay outside the
production decision path.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
data/generalization/v1/           three generated offline evaluation sets
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  submitted Agent adapter for the team pipeline
evaluator/local_evaluator.py      public-set simulator and scorer
scripts/prepare_catalog.py        validate and unpack a local catalog archive
src/pipeline/                    integrated production pipeline
src/attribute.py                  shared shopping-attribute contract
src/item.py                       shared Item and candidate contracts
src/state/                        intent routing and conversation state
src/retrieval/                    routed BM25 and optional Dense experiments
src/reranking/                    candidate reranking module and tests
src/dialogue/                     clarification-question module and tests
examples/reranker_demo.py         runnable Reranking integration example
```

## Submission policy

See [`docs/submission_rules.md`](docs/submission_rules.md),
[`docs/competition_specification.md`](docs/competition_specification.md), and
[`docs/agent_api_contract.json`](docs/agent_api_contract.json) for the
participant rules and interface. This repository contains participant code;
organizer-only judging controls and private evaluation data are not included.

Submit the Agent entry point, required local helper modules, dependency/setup
instructions, and a method/limitations report with the chosen runtime mode,
latency, token usage, and estimated cost. Do not submit secrets or depend on
undeclared external services.

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
