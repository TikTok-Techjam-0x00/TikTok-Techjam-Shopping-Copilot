# Technical Report

> 🧠 From conversation State to evidence-aware recommendations—one turn at a
> time.

[![System](https://img.shields.io/badge/System-sota--2.2-4f46e5?style=flat-square)](README.md)
![Public Sessions](https://img.shields.io/badge/Public_Sessions-200-2563eb?style=flat-square)
![HitRate@10](https://img.shields.io/badge/HitRate%4010-0.995-059669?style=flat-square)
![MRR](https://img.shields.io/badge/MRR-0.960833-059669?style=flat-square)
![MTTC](https://img.shields.io/badge/MTTC-2.23-f59e0b?style=flat-square)

[🎬 Watch Demo](https://youtu.be/tSkZy2qN8lI) ·
[🏠 Back to README](README.md) · [📊 Jump to Results](#-6-evaluation-results) ·
[⚠️ Jump to Limitations](#️-8-limitations)

This report describes the implementation shipped in this repository. It focuses
on what the Agent does on every conversation turn and how its components react
to changes in shopping intent and constraints. Algorithmic derivations are kept
brief; the corresponding source files are linked where useful.

## 🎯 1. Problem

The Agent receives a user profile and a conversation of at most ten turns. On
each turn it must return:

- a customer-facing message;
- one allowed `ask_attribute`, or `null`;
- an ordered list of at most ten product `parent_asin` values; and
- reported prompt and completion token usage.

The target product and hidden intent card are never supplied to the Agent. The
system must infer whether the user is buying or browsing, accumulate only the
requirements disclosed so far, retrieve from a 50,000-product catalog, and
balance early recommendation quality against the value of asking another
question. A successful session is one in which the target appears in the first
ten valid recommendations on any turn.

The implementation is designed around three practical goals:

1. preserve conversational constraints without leaking superseded preferences;
2. keep the ranking path deterministic, local, and inexpensive; and
3. explore new candidates in later turns instead of repeating an exhausted
   first page.

## 🏗️ 2. System Architecture

The official entry point is `starter.agent.Agent`. It owns one `Pipeline`, and
the Pipeline stores an isolated `ShoppingState` and recommendation memory for
each session.

```text
Agent.reset(session_id, user_profile)
                  |
                  v
          Per-session ShoppingState

Agent.respond(user message, turn, top_k)
                  |
                  v
       1. Intent and slot update
                  |
                  v
       2. State-derived query
                  |
                  v
       3. Intent-routed Retrieval ------ optional turn-8 Dense residual
                  |                                  |
                  +------------- Top 100 ------------+
                                    |
                                    v
                       4. Evidence Coverage Reranker
                                    |
                       unseen ordered recommendations
                                    |
                                    v
                        5. Clarification decision
                                    |
                                    v
                  message + ask_attribute + products
```

Every call to `respond` runs the full sequence again. Retrieval and Reranking
are therefore **per-turn operations**, not one-time initialization steps. The
updated State changes the next query, candidate pool, ranking evidence, and
question choice.

The major implementation boundaries are:

| Component | Responsibility | Main implementation |
|---|---|---|
| Agent adapter | Match the evaluator interface and configure optional Qwen State resolution | `starter/agent.py` |
| Pipeline | Coordinate one complete turn and maintain session-local memory | `src/pipeline/pipeline.py` |
| State | Store intent, constraints, provenance, epochs, history, and asked attributes | `src/state/` |
| Retrieval | Build a State query and return up to 100 BM25/Dense candidates | `src/retrieval/` |
| Reranking | Reorder candidates by catalog evidence for active requirements | `src/reranking/evidence.py` |
| Dialogue | Select an open or attribute-specific clarification question | `src/dialogue/` |

Catalog loading and index construction happen when the Agent is initialized.
Conversation state is then mutated only within its session ID; one user's
preferences cannot enter another user's State.

## 🧠 3. State & Intent Management

### 3.1 🧾 State representation

`ShoppingState` records:

- current `buying` or `browsing` intent and its confidence/source;
- hard constraints and soft constraints;
- attributes for which the user declared no preference;
- explicitly rejected values;
- user-message history and previously asked attributes;
- intent transitions, boundary detection, and optional semantic-fallback
  diagnostics; and
- a `constraint_epoch` plus per-constraint provenance (`source_turn`, epoch,
  confidence).

The canonical attributes are `category`, `material`, `color`, `size`, `style`,
`brand`, `budget`, `feature`, `use_case`, and `other`.

### 3.2 🔄 What happens on each turn

For every message, the State manager performs the following sequence:

1. A deterministic intent classifier produces an intent, confidence, evidence,
   conflict flag, and override flag.
2. Rule-based slot extraction converts explicit requirements into a normalized
   `StateUpdate`.
3. If an optional semantic resolver is configured, it is considered only for
   unresolved cases such as low intent confidence, conflicting signals,
   context-dependent references, comparisons, alternatives, or an override
   with no resolved attributes.
4. Semantic output must match the structured schema and confidence thresholds.
   Rule-extracted facts take precedence when rule and model output are merged.
5. History-aware smoothing prevents weak evidence from flipping an established
   intent. An explicit change uses a lower transition threshold than an
   unprompted change.
6. The normalized update is applied, provenance is recorded, and the raw user
   message is appended to history.

The optional Qwen path is therefore a bounded fallback for ambiguity, not the
primary ranking mechanism. With no `DASHSCOPE_API_KEY`, all State updates remain
local and rule-based.

### 3.3 🧭 Responses to different State changes

| State event | Concrete system response |
|---|---|
| New hard constraint | Replace the previous value for that attribute; use weight 3 in Reranking |
| New soft constraint | Merge it with existing soft preferences; use weight 2 in Reranking |
| No preference | Remove positive hard/soft values for the attribute and exclude it from future clarification |
| Rejected value | Store it separately; remove a conflicting positive hard value |
| Buying intent | Use the detailed Buying BM25 field weights |
| Browsing intent on turn 1 | Use a category-heavy, concise Browsing route |
| Browsing after turn 1 | Fall back to the more precise Buying route as more information becomes available |
| Explicit intent/category override | Start a new constraint epoch, reset asked-attribute and recommendation memory, and replace obsolete category/use-case constraints |
| Change to browsing | Clear hard constraints and rejected values so the previous purchase target does not dominate exploration |
| Boundary disclosure | Record which attributes crossed a stated boundary and expose this in debug State |
| Ambiguous reference with API enabled | Request a structured semantic resolution; validate it before merging |
| Ambiguous reference without API | Retain the best rule/history-based interpretation and continue deterministically |

An override does not blindly erase every historical field. Provenance keeps old
records available for diagnostics, while retrieval excludes stale soft
preferences from earlier epochs. This separates audit history from active
ranking context.

### 3.4 🔎 Query construction from State

After the update, Retrieval builds a fresh lexical query from active State
rather than concatenating the whole conversation. Category terms receive first
priority, followed by values introduced on the override turn, values from the
current epoch, and still-valid older hard constraints. Stale soft preferences
from an earlier epoch and attributes marked as no-preference are omitted.

Only when State contains no useful values does the current user message become
the fallback query. CJK text is removed before keyword or Dense retrieval
because the released catalog and evaluation constraints are English.

## 🔍 4. Retrieval and Reranking

### 4.1 📚 Retrieval on every turn

The primary retriever is SQLite FTS5 BM25 over structured product fields. It
uses separate weight profiles:

| Field | Buying weight | Browsing weight |
|---|---:|---:|
| Title | 6.0 | 5.0 |
| Categories | 4.0 | 6.0 |
| Features | 2.5 | 2.5 |
| Derived attributes | 2.5 | 2.5 |
| Details | 2.5 | 2.5 |
| Store | 0.75 | 1.5 |
| Description | 0.5 | 1.0 |

Buying emphasizes product identity and disclosed constraints while reducing
noise from merchant and long-description text. The first browsing turn gives
more weight to taxonomy so a broad request can establish the product area.

The production turn schedule deliberately changes the rank window:

| Turn | Candidate cohort passed to Reranking |
|---:|---|
| 1–6 | BM25 ranks 1–100 |
| 7 | BM25 ranks 101–200 |
| 8 | Residual page at ranks 201–300; optional lexical-gated Dense supplementation |
| 9 | 50 candidates from ranks 1–50 plus 50 from ranks 401–450 |
| 10 | BM25 ranks 301–400 |

This schedule gives early turns the strongest lexical results and later turns
controlled breadth. If `retrieval_pool_size` is configured above 100, the
Pipeline instead retrieves that larger first page, removes already shown
products, and takes the first 100 remaining candidates.

### 4.2 🧬 Optional Dense residual

Dense retrieval is used only when all of the following are available:

- a compatible 50,000 × 256 `text-embedding-v4` cache;
- an embedding API key and endpoint configuration; and
- a successful query-embedding call.

On turn 8, the implementation protects the first ten lexical candidates of the
selected residual page and permits at most ten semantic supplements. A Dense
candidate must also occur within the deeper top-1,000 BM25 result and must have
a lexical rank of at least 301. This lexical gate prevents unrelated dense-only
matches from displacing the established BM25 head. Missing files, incompatible
metadata, provider errors, or absent credentials produce an automatic BM25-only
fallback.

### 4.3 🏆 Evidence Coverage Reranker

Retrieval answers “which products are plausible?” The Reranker answers “which
of these candidates best supports everything the user has disclosed?” It
extracts atomic evidence fragments from active hard and soft constraints, then
measures their token coverage against title, taxonomy, features, attributes,
details, description, and store evidence.

The primary score is IDF-weighted requirement coverage. Hard fragments have
weight 3, soft fragments weight 2, and category weight 1 so a generic taxonomy
match cannot overwhelm specific needs. Phrase matches add a bounded bonus. The
following signals are then used in deterministic tie-breaking:

1. explicit budget compliance and, for “around” requests, price proximity;
2. number of completely matched requirements;
3. exact catalog-field evidence;
4. category evidence in identity fields;
5. popularity and average rating;
6. the original retrieval score and order.

Numeric dimensions are not treated as prices unless the conversation contains
explicit monetary context. Products with missing prices are treated as unknown,
not automatically over budget.

If State has no active fragments and no budget, Reranking preserves retrieval
order. Otherwise all 100 candidates are reranked. The Pipeline then filters out
products already shown in the current constraint epoch. Turns 1 and 2 expose
only the strongest unseen product; from turn 3 onward, up to the requested
`top_k` products are returned. Continuing the conversation consequently acts
as implicit negative feedback for previously displayed products without
rewriting the user's explicit constraints.

## 💬 5. Clarification Strategy

Clarification is computed after Retrieval and Reranking on every turn, and a
question is returned alongside the current recommendations.

### 5.1 🌱 First three turns

Turns 1–3 intentionally use open questions with `ask_attribute="other"`:

1. must-have details;
2. important features or preferences; and
3. final requirements or deal-breakers.

This lets the evaluator disclose constraints whose catalog field is not known
in advance. It also explains why only one recommendation is exposed on turns 1
and 2: the system avoids spending all Top-10 slots before the user has revealed
the missing requirements. Turn 3 already returns up to `top_k` while asking the
last open question.

### 5.2 🎛️ What happens after turn 3

Turns 4–9 switch to the attribute-specific 3B policy; they do **not** stop
clarifying immediately. The selector first excludes attributes that are already
known, already asked in the current epoch, or explicitly marked no-preference.
If category is still unknown, it is asked first. Otherwise each remaining
attribute receives:

- a fixed semantic priority;
- a profile-tag boost when the user's aggregate profile maps to that attribute;
- a dynamic question-value term based on coverage and diversity in the current
  Top-100 candidates; and
- a late-turn boost for feature, size, material, and budget from turn 4 onward.

The dynamic term favors questions that the catalog can answer and whose values
meaningfully separate the current candidates. Suggested options are drawn from
the most common values in those candidates, not from hidden labels. Asked
attributes are written back into State so they are not repeated during the same
constraint epoch.

An explicit override starts a new epoch and clears the asked-attribute set, so
questions relevant to the new shopping goal can be asked again. Turn 10 returns
recommendations only: `ask_attribute` is `null`, and no further clarification
is attempted.

## 📊 6. Evaluation Results

The official evaluator uses 200 public sessions, exact `parent_asin` matching,
`top_k=10`, a maximum of ten turns, and a miss value of 11 turns. Metrics are:

```text
HitRate@10    = successful sessions / session count
MRR           = mean reciprocal rank of the first target hit
MTTC          = mean first-hit turn, assigning 11 to misses
Efficiency    = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
```

The retained weak BM25 baseline and the current SOTA 2.2 production runs are:

| System | HitRate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Weak BM25 baseline | 0.125000 | 0.068034 | 9.81000 | 0.119000 | 0.106710 |
| Production Agent, no API key | 0.995000 | 0.960833 | 2.23000 | 0.877000 | 0.961150 |
| Production Agent, API enabled | 0.995000 | 0.960833 | 2.23000 | 0.877000 | 0.961150 |

The latest API-enabled acceptance run produced the following scenario-level
results through the unmodified official evaluator:

| Scenario | Sessions | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 0.987500 | 0.965625 | 1.875000 |
| Browsing | 80 | 1.000000 | 0.963542 | 2.037500 |
| Intent Override | 30 | 1.000000 | 0.927778 | 3.633333 |
| Boundary | 10 | 1.000000 | 1.000000 | 2.400000 |
| **Overall** | **200** | **0.995000** | **0.960833** | **2.230000** |

Final acceptance summary:

- **TechnicalScore:** 0.961150
- **Efficiency:** 0.877000
- **Evaluator-reported token usage:** 2,099 prompt + 271 completion = 2,370
- **Repository unit tests:** 8/8 passed

The production value is the repository's documented Public-200 reference run;
it is not a claim about hidden-test performance. Reproduction commands and data
preparation checks are provided in `README.md`.

## ⚡ 7. Runtime, Token Usage and Cost

Complete Public-200 runs were measured including Agent initialization. The
offline reference took about 51 seconds. The final API-enabled acceptance run
on 2026-08-31 used Python 3.11.9, the shipped 50,000 x 256 LFS vector cache,
and the default Singapore DashScope endpoints.

| Mode | Network during evaluation | Evaluator-reported token usage | Approximate direct model cost |
|---|---|---:|---:|
| No API key | None after installation/data preparation | 0 | USD 0 |
| API key and compatible Dense cache | Two `qwen-plus` State calls and one turn-8 `text-embedding-v4` query call | 2,099 input + 271 output = 2,370 | USD 0.001134 |

Provider usage in the final API-enabled run was:

- `qwen-plus`: 2 calls, 2,006 input tokens and 271 output tokens;
- `text-embedding-v4` (256 dimensions, `text_type=query`): 1 call and 93
  input tokens, with no completion tokens.

The cost estimate uses the Singapore list prices available on 2026-08-31:
USD 0.40 per million input tokens and USD 1.20 per million output tokens for
non-thinking `qwen-plus`, plus USD 0.07 per million input tokens for
`text-embedding-v4`. Free quotas and promotions are excluded. The estimate also
excludes the historical one-time cost of building the reusable 50,000-product
embedding cache because catalog embeddings are not regenerated during
evaluation. Pricing references are the Alibaba Cloud Model Studio
[`qwen-plus` pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
and [`text-embedding-v4` API documentation](https://www.alibabacloud.com/help/en/model-studio/text-embedding-synchronous-api).

BM25, Evidence Coverage Reranking, and clarification are local and consume no
model tokens. The Agent now reports the provider-returned State and query-
embedding usage on each turn. The official evaluator sums those per-turn values;
no evaluator code or model output is modified. Turns that make no model call
correctly report zero usage.

The submission does not require network access for its reliable default path.
If credentials, the compatible vector cache, network access, or the provider
are unavailable, it automatically falls back to local BM25. The offline path
requires neither model calls nor the optional LFS vectors after installation
and catalog preparation.

## ⚠️ 8. Limitations

- The deterministic slot and intent rules cover common English shopping
  language but can miss unusual paraphrases, long-range references, sarcasm,
  and multilingual input. CJK text is intentionally removed from retrieval.
- Semantic State resolution and Dense query embedding require external
  credentials, network availability, and compatible provider behavior. The
  robust no-key path sacrifices some semantic understanding.
- Dense retrieval is deliberately limited to a lexical-gated turn-8 residual;
  it is not a full hybrid fusion system and cannot rescue every lexical miss.
- Evidence Coverage is lexical. Synonyms not normalized by State or represented
  in catalog text may receive insufficient credit.
- Rejected values are maintained in State, but the production evidence
  reranker's main score is driven by positive hard/soft fragments and budget;
  negative-constraint handling is less expressive than positive matching.
- The fixed late-turn retrieval windows improve breadth for the ten-turn
  evaluator but are heuristic and may not be optimal for conversations with a
  different length or catalog distribution.
- Recommendation memory treats continued conversation as implicit rejection.
  A user who still likes an earlier product but asks an unrelated question may
  not see it again until an override starts a new epoch.
- The runtime figures are end-to-end local observations and do not separate
  index construction, per-turn latency, or API latency.
- Public-set performance can guide engineering but does not guarantee the same
  result on hidden sessions or future catalog versions.

## 👥 9. Team Contributions

0x00 Shopping Copilot was developed as a collaborative team effort. Rather than assigning isolated modules to individual members, we worked through shared design reviews, implementation, evaluation, and integration across the full system.

Team contributions covered:

- **System architecture and agent design:** defining the end-to-end conversational shopping pipeline and interfaces between State, Retrieval, Reranking, Dialogue, and Evaluation.
- **Conversational state and intent handling:** designing constraint tracking, intent classification, override handling, and clarification logic.
- **Retrieval and ranking:** developing and iterating on BM25 retrieval, semantic retrieval experiments, evidence-aware reranking, and candidate exploration strategies.
- **Evaluation and experimentation:** integrating the official evaluator, running controlled experiments, analyzing failure cases, and iterating toward the final SOTA 2.2 system.
- **Integration and reliability:** connecting components into the official `Agent` interface, validating reproducibility, handling fallbacks, and preparing the final submission.
- **Demo and communication:** building the demonstration frontend, technical documentation, architecture presentation, and project video.

Responsibilities evolved throughout the hackathon, and team members frequently collaborated across component boundaries through code review, debugging, experimentation, and integration. The final system therefore reflects shared ownership of the overall architecture and engineering outcome.
