# TikTok TechJam Shopping Copilot

Team 0x00's final Agent for TechJam 2026 Track 4, based on `sota-2.2`
(`d5ac6e9`).
This submission keeps the core Agent, the presentation frontend, and the
official participant evaluation toolkit and documents.

## Agent entry point

The official evaluator imports `Agent` from `starter/agent.py`:

~~~python
from starter.agent import Agent

agent = Agent(catalog_path="data/catalog.jsonl")
agent.reset(session_id="demo", user_profile={})
response = agent.respond(
    session_id="demo",
    user_message="I am looking for black running shoes.",
    turn=1,
    top_k=10,
)
~~~

`response` contains `message`, `ask_attribute`, `recommendations`
(a list of `{"parent_asin": "..."}` objects), and `usage`.
Call `reset` once per session and preserve the same Agent for subsequent
turns. The Agent never receives target labels or hidden intent cards.

## Installation and catalog

Python 3.10+ is supported; local validation used Python 3.11.4.
Run commands from the repository root.

~~~bash
python -m venv .venv
~~~

Activate with `.venv\Scripts\Activate.ps1` on Windows or
`source .venv/bin/activate` on macOS/Linux, then install:

~~~bash
python -m pip install -r requirements.txt
~~~

Download `catalog.jsonl.gz` from the
[official Participant Kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participantkit).
Prepare the frozen 50,000-product catalog:

~~~bash
python scripts/prepare_catalog.py --archive /path/to/catalog.jsonl.gz
~~~

The script checks both archive and catalog SHA256, writes
`data/catalog.jsonl`, and refuses to overwrite a different existing file.
It does not download data. The full upstream Amazon dataset is not needed.

## Production workflow

1. **State:** maintain buying/browsing intent, disclosed constraints,
   no-preference fields, rejected values, and the current constraint epoch.
2. **Retrieval:** use intent-routed, field-weighted BM25, with deeper rank
   windows in late turns and an optional turn-8 semantic residual.
3. **Ranking:** apply `EvidenceCoverageReranker` directly to product evidence.
   Ranking is local and does not call an LLM.
4. **Dialogue:** ask clarification questions while returning recommendations.
   Turns 1-2 expose one product each; later turns expose up to `top_k`.
   Already-shown products are skipped within an epoch; an override resets
   recommendation memory.

Shared contracts are `Item`, `Candidate(item=...)`, and
`RankedCandidate(item=...)`. Canonical shopping attributes are:
`category, material, color, size, style, brand, budget, feature, use_case, other`.

## Runtime modes and external services

The original SOTA 2.2 automatic behavior remains:

| Configuration | Retrieval | State |
|---|---|---|
| No API key | Local BM25 | Local rule-based updates |
| API key, no compatible vector cache | BM25 fallback | Optional Qwen semantic resolution |
| API key and compatible needs cache | BM25 plus turn-8 Dense residual | Optional Qwen semantic resolution |

A local `.env` or process environment can change the execution path. Copy
`.env.example` only when intentionally enabling external services. Never
commit a real API key.

For Dense, retrieve the only shipped vector cache:

~~~bash
git lfs pull --include="artifacts/retrieval/dense/text-embedding-v4__dense_needs_v1__d256/embeddings.npy"
~~~

The cache contains `embeddings.npy`, `manifest.json`, and
`parent_asins.json`: `text-embedding-v4`, `dense_needs_v1`, 256 dimensions.
A Git LFS pointer alone is not a usable matrix. On turn 8, semantic candidates
must also be supported by deeper BM25 ranks. Missing/incompatible cache,
credentials, or a Dense provider error falls back to BM25.

The no-key path needs no vectors, model API, or network after installation
and data preparation. To force this mode despite local `.env`, set the key
to an empty string **before importing the Agent**:

~~~python
import os
os.environ["DASHSCOPE_API_KEY"] = ""
from starter.agent import Agent
~~~

Optional embedding and State calls require network access and may incur
costs and timeout/retry latency. The Agent response reports zero ranking tokens;
it does not aggregate embedding or State-model usage. Do not
interpret zero reported tokens in an API-enabled run as zero provider usage.

## Local verification

The official local evaluator and public 200 sessions are retained for
acceptance checks. The evaluator has local progress-display and `--limit`
CLI additions; its simulation and scoring logic are unchanged.
Run without API access:

~~~bash
python -c "from pathlib import Path; Path('artifacts/validation').mkdir(parents=True, exist_ok=True)"
python -c "import os, runpy; os.environ['DASHSCOPE_API_KEY']=''; runpy.run_module('evaluator.local_evaluator', run_name='__main__')" --output artifacts/validation/public200.json
~~~

Use `--limit 4` for a smoke test. Output is ignored by Git.
No-key Public-200 reference: **TechnicalScore 0.961150**, **HitRate@10
0.995000**, **MRR 0.960833**, **MTTC 2.23000**. Local full-run time is about
51 seconds including initialization; this is not a controlled latency
benchmark or a guarantee of hidden-test performance.

The organizer's evaluator tests are also retained:

~~~bash
python -c "import os, runpy; os.environ['DASHSCOPE_API_KEY']=''; runpy.run_module('unittest', run_name='__main__')" tests.test_evaluator
~~~

## Presentation frontend

The frontend is optional and is not imported by the Agent or evaluator.
After preparing the catalog, install its dependencies and start the local
server from the repository root. This command forces the same no-key mode
as the reference evaluation:

~~~bash
python -m pip install -r frontend/requirements.txt
python -c "import os, uvicorn; os.environ['DASHSCOPE_API_KEY']=''; uvicorn.run('frontend.server:app', host='127.0.0.1', port=8000)"
~~~

Open `http://127.0.0.1:8000`. The Agent, product-detail adapter, and Developer
Mode share one catalog. The production indexes are built at startup; the
developer-only BM25 index is built on its first retrieval step.

- **Demo Mode:** uses the production `starter.Agent`. Select a public sample,
  then use **Start Sample**, **Next Turn**, or **Auto Run**. The evaluator
  controller generates customer replies and stops on a hit or the turn limit.
- **Developer Mode:** an isolated, step-by-step component inspector. Load a
  turn, run individual stages or all remaining stages, then commit the turn.
  This view uses `SimpleReranker`, not the production Evidence Coverage
  ranking path; it is not an exact trace of SOTA 2.2 inference.

Recommendations are enriched from the real catalog. Ground-truth labels stay
in the frontend evaluation controller and are never supplied to the Agent.
Demo and developer sessions are separate, stored in memory, and lost when
the server restarts. The local server is intended for presentation, not
public deployment.

The HTTP chat endpoints (`POST /api/session`, `POST /api/chat`) remain
available; the presentation UI uses the evaluator-driven flow above.

## Submission files

| Path | Purpose |
|---|---|
| `starter/agent.py` | Official Agent adapter |
| `src/pipeline/` | Production orchestration |
| `src/state/` | State and optional semantic resolution |
| `src/retrieval/` | Retrieval and vector-cache support |
| `src/reranking/` | Evidence Coverage ranking and the developer inspector's rule-based ranker |
| `src/dialogue/` | Clarification decisions |
| `src/item.py`, `src/attribute.py` | Shared product and attribute contracts |
| `requirements.txt`, `.env.example` | Installation and optional service configuration |
| `scripts/prepare_catalog.py` | Local catalog preparation |
| `artifacts/retrieval/dense/*dense_needs_v1*/` | Optional SOTA 2.2 vector cache |
| `frontend/` | Demo UI, HTTP adapter, and component inspector |
| `data/public_set.jsonl` | Official 200-session public test set |
| `evaluator/`, `tests/test_evaluator.py` | Official local evaluator and its supplied tests |
| `docs/` | Official specification, submission rules, API contract, evaluation config, and baseline results |
| `data/README.md`, `DATA_ATTRIBUTION.md` | Data instructions and attribution |

The downloaded catalog, local environment, and generated evaluation outputs
are ignored by Git. The retained vector-cache files are already tracked;
the matrix is stored with Git LFS.

## Data attribution

The catalog is derived from **Amazon Reviews 2023**, McAuley Lab, UCSD,
category `Clothing_Shoes_and_Jewelry`; products are joined by `parent_asin`.
The competition modality is text and structured metadata only.
Follow the source dataset's applicable terms; the competition organizer does
not claim ownership of the underlying Amazon content. See
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) for the retained attribution notice.
