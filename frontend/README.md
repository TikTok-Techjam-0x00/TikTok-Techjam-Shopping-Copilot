# Shopping Copilot Hackathon Frontend / 购物助手黑客松前端

A desktop-first demo interface for presenting the repository's existing multi-turn Shopping Copilot Agent. The UI is designed for a 16:9 hackathon presentation or screen recording and shows the live conversation, real catalog recommendations, and observable Agent state.

## Architecture

```text
frontend/index.html + frontend/styles.css + frontend/app.js
                            ↓
                    frontend/server.py
                            ↓
                      starter.Agent
                            ↓
                    Existing Pipeline
                            ↓
          State / Retrieval / Reranking / Dialogue
```

`frontend/server.py` is only an HTTP/session adapter. It imports the existing Agent with:

```python
from starter.agent import Agent
```

It calls `Agent.reset()` once when a browser session is created, calls `Agent.respond()` for turns 1–10, and reads the structured state through `Agent.get_state()`. It does not contain or duplicate shopping intelligence.

## Product enrichment

The official Agent response contains recommendation `parent_asin` values only. The server loads the existing catalog:

```python
from src.retrieval.catalog import Catalog
catalog = Catalog.load("data/catalog.jsonl")
```

Each recommended ASIN is looked up in this catalog and serialized with the existing `Item.to_dict()` method. The browser receives real titles, prices, stores, ratings, categories, features, descriptions, and details. Because the catalog has no image contract, the UI intentionally uses a neutral product placeholder rather than fabricated images.

## Dependencies

- Python 3.10+
- Existing repository environment and catalog at `data/catalog.jsonl`
- FastAPI
- Uvicorn

All demo-specific Python dependencies are declared in `frontend/requirements.txt`.

## Setup

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r frontend\requirements.txt
```

Or, with an activated virtual environment:

```powershell
pip install -r frontend/requirements.txt
```

## Start the demo

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn frontend.server:app --host 127.0.0.1 --port 8000
```

For local development with reload:

```powershell
.\.venv\Scripts\python.exe -m uvicorn frontend.server:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Initial startup loads the 50,000-product catalog and builds the existing retrieval index, so allow a few seconds before the page becomes available.

## Evaluator-driven demo lifecycle

The presentation UI does not require a presenter to role-play the customer. It selects a public evaluator sample and reuses the official evaluator helpers to prepare the initial message, follow-up customer replies, boundary behavior, and intent override message.

- **Start Sample** creates a fresh Agent session using the selected sample's anonymized profile.
- **Next Turn** runs one complete evaluator turn and visualizes the generated user message, Agent response, recommendations, and State.
- **Auto Run** advances until the target is hit or the official 10-turn limit is reached. It can be paused.
- Ground truth is retained only by the frontend evaluator controller for stop/hit reporting. It is never passed to the Agent.
- The same Agent session is preserved across generated messages and reset only when another sample is started.

## Endpoints

- `GET /` — demo page
- `GET /api/health` — server and catalog readiness
- `POST /api/session` — create/reset a browser demo session
- `POST /api/chat` — send one multi-turn message
- `GET /api/eval/samples` — list safe public sample metadata
- `POST /api/eval/session` — start an evaluator-driven Demo session
- `POST /api/eval/next` — run one complete evaluator turn

## Presentation notes

- Use a 1920×1080 browser window for the intended layout.
- The pipeline visualization is architectural, not per-stage telemetry.
- Developer Data is collapsed by default and contains only observable Agent response/state JSON, never chain-of-thought.

## Developer Mode

Developer Mode is an isolated pipeline inspector for engineering and integration work. It does not call `Agent.respond()` and then reconstruct fake diagnostics. Instead, `frontend/trace_runner.py` calls the same existing components in the same order as the production Pipeline:

```text
Input
  → src.state.update_state
  → src.state.retrieval_query (or sanitize_retrieval_text fallback)
  → src.retrieval.Retriever.retrieve
  → src.reranking.SimpleReranker.rerank
  → src.dialogue.decide_ask + record_asked_attribute
  → src.reranking.recommendations_from_ranking
  → Official AgentResponse
```

Demo and Developer sessions are fully separate. Demo Mode uses `starter.Agent`; Developer Mode owns a separate `ShoppingState`, Retriever, Reranker, active trace, and committed turn history. A partially executed developer turn cannot mutate the Demo session.

### Developer tracing lifecycle

1. Switch to **Developer Mode** and create an isolated developer session.
2. Select a public sample. The evaluator controller prepares the next user message automatically.
3. Click **Load Evaluator Turn**. This stores the prepared message only; it does not run pipeline stages.
4. Click a ready stage or **Run Next Step** to execute one real component.
5. Use **Run All Remaining** to execute all pending stages sequentially while retaining every intermediate result.
6. Click completed stages to inspect cached inputs, outputs, raw JSON, state differences, candidates, scores, ranks, matched attributes, violations, and dialogue decisions. Completed stages are not rerun.
7. Use **Restart Turn** to discard the active working state and explicitly restart that turn.
8. After Final Response, click **Commit & Load Next**. The server atomically commits the working state, prepares the next customer reply from the real `ask_attribute`, and creates that next turn's Input stage. **Active Turn** confirms which turn is ready. **Run Current Turn** runs only the remaining stages of that active turn. The official evaluator still stops immediately when its target product is hit.
9. The result badge shows **Turn N · Not Hit Yet**, **Target Hit · Rank N**, or **Max Turns · No Hit** after each committed evaluator turn.
9. Use Turn History to inspect committed traces without rerunning them.

Execution duration is measured with `time.perf_counter()` immediately around each real component call and labeled **Local execution time**. These values are diagnostics, not official benchmark latency.

### Trace API endpoints

- `POST /api/dev/session` — create an isolated developer session
- `POST /api/dev/turn` — store a new turn input without running pipeline stages
- `POST /api/dev/stage` — execute one named next stage
- `POST /api/dev/next` — execute only the next pending stage
- `POST /api/dev/all` — execute all remaining stages in order
- `POST /api/dev/restart` — explicitly restart the active turn
- `POST /api/dev/commit` — commit completed working state for the next turn
- `GET /api/dev/trace/{session_id}/{turn}` — retrieve a stored turn trace
- `POST /api/dev/scenario` — create an isolated trace session from a public evaluator sample

## Limitations

- Demo and trace sessions live in server memory and disappear when the server restarts.
- The existing SQLite retriever is thread-bound, so frontend Agent/Trace operations execute under a lock in the Uvicorn event-loop thread.
- Developer timings include local wrapper call duration and are not competition latency measurements.
- Developer Mode exposes structured inputs, outputs, ranks, scores, and errors only. It does not expose or fabricate chain-of-thought.
- Trace parameters are read-only and use evaluator defaults (`top_k=10`, retrieval `k=100`).

---

# 中文说明

这是一个面向 16:9 黑客松展示与录屏的 Shopping Copilot 前端，同时提供干净的 **演示模式** 和可逐步检查真实流水线的 **开发者模式**。

## 架构

```text
                         ┌─ 演示模式 ───────────────┐
浏览器 HTML / CSS / JS ──┤                          │
                         └─ 开发者模式 ─────────────┘
                                      ↓
                            frontend/server.py
                              ↓              ↓
                         Demo API         Trace API
                              ↓              ↓
                       starter.Agent   frontend/trace_runner.py
                              ↓              ↓
                              └── 现有核心组件 ──┐
                                                ↓
                              State / Retrieval / Reranking / Dialogue
```

`frontend/server.py` 只负责 HTTP、内存 session 和 catalog enrichment，不包含购物算法。所有购物逻辑仍来自仓库现有实现。

## 依赖与安装

- Python 3.10+
- 现有项目环境
- `data/catalog.jsonl`
- FastAPI
- Uvicorn

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r frontend\requirements.txt
```

已激活虚拟环境时也可运行：

```powershell
pip install -r frontend/requirements.txt
```

## 启动

```powershell
.\.venv\Scripts\python.exe -m uvicorn frontend.server:app --host 127.0.0.1 --port 8000
```

开发热更新模式：

```powershell
.\.venv\Scripts\python.exe -m uvicorn frontend.server:app --reload
```

浏览器访问：

```text
http://127.0.0.1:8000
```

启动时会加载 50,000 个商品并建立现有检索索引，因此页面可用前需要等待几秒。

## 演示模式

演示模式用于黑客松展示和录屏，界面重点是：

- 完整多轮对话；
- 真实 catalog 商品推荐；
- Agent 当前结构化 State；
- 简洁的 Pipeline 架构展示；
- New Session 生命周期。

演示者不需要扮演顾客输入或回答问题。界面选择一个 public sample，并复用官方 evaluator helper 自动准备 initial message、追问回复、Boundary 行为和 Intent Override 消息。

- **Start Sample**：使用样本的匿名画像创建新 Agent session。
- **Next Turn**：完整执行一轮 evaluator，并展示自动用户消息、Agent 回复、推荐与 State。
- **Auto Run**：自动运行到命中目标或官方 10 轮上限，支持暂停。
- Ground truth 只由 frontend evaluator controller 用于判断停止和命中，不会传给 Agent。
- 整个自动对话复用同一个 Agent session，选择新样本时才 reset。

## 商品详情 enrichment

Agent 官方输出只包含 `parent_asin`。服务器使用：

```python
from src.retrieval.catalog import Catalog
```

根据 ASIN 查找真实 `Item`，再使用现有 `Item.to_dict()` 返回标题、价格、商店、评分、类别、features、description 和 details。Catalog 不包含图片，因此 UI 使用中性占位图，不伪造商品照片。

## 开发者模式

开发者模式用于集成、调试和理解单轮 Pipeline。它不会先运行 `Agent.respond()` 再伪造中间数据，而是由 `frontend/trace_runner.py` 按生产 Pipeline 的真实顺序调用现有组件：

```text
Input 输入
  → src.state.update_state
  → src.state.retrieval_query
    或 sanitize_retrieval_text fallback
  → src.retrieval.Retriever.retrieve
  → src.reranking.SimpleReranker.rerank
  → src.dialogue.decide_ask
  → src.dialogue.record_asked_attribute
  → src.reranking.recommendations_from_ranking
  → 官方 AgentResponse
```

开发者模式拥有独立的 `ShoppingState`、Retriever、Reranker、工作 trace 与历史记录，不与 Demo Agent session 混用。未完成的开发者 turn 不会污染演示模式。

### 逐步 Trace 使用方法

1. 切换到 **Developer Mode**，创建独立开发 session。
2. 选择 public sample，evaluator controller 会自动准备下一条用户消息。
3. 点击 **Load Evaluator Turn**，此时只载入官方模拟消息，不执行其他阶段。
4. 点击当前可执行阶段，或点击 **Run Next Step**，只执行下一个真实 Python 组件。
5. 点击 **Run All Remaining**，按顺序执行所有剩余阶段，并保存每个中间结果。
6. 点击已完成阶段可以查看缓存的输入、输出、Raw JSON、State diff、候选、分数、排名变化、matched、violation 和 Dialogue Decision，不会重新运行。
7. **Restart Turn** 会明确丢弃当前工作状态并重新开始该轮。
8. Final Response 完成后点击 **Commit & Load Next**。服务器会原子地提交工作 State，根据真实 `ask_attribute` 自动准备 evaluator 用户回复，并创建下一轮 Input。顶部的 **Active Turn** 会明确显示当前已载入的轮次；**Run Current Turn** 只运行当前轮剩余阶段。若目标商品已命中，官方 evaluator 会按真实规则立即结束场景。
9. 每次提交 evaluator 轮次后，结果徽标会显示 **Turn N · Not Hit Yet**、**Target Hit · Rank N** 或 **Max Turns · No Hit**。
9. Turn History 可以查看旧轮次所有 trace，不会重新执行旧 turn。

### 各阶段展示内容

- Input：session ID、turn、用户画像、消息、上一轮 asked attribute。
- State Update：State Before、State After、真实字段差异。
- Retrieval Query：真实 query、来源、是否使用 fallback。
- Retrieval：真实 Candidates100、分数、排名、商品详情和显示过滤。
- Reranking：真实 Candidates10、retrieval/rerank 排名变化、matched 和 violation。
- Dialogue：真实 `ask_attribute`、message、输入 State 和候选数量。
- Final Response：与官方一致的 response JSON。

每个阶段使用 `time.perf_counter()` 包围真实函数调用，显示 **Local execution time / 本地执行时间**。它只用于开发诊断，不是官方 benchmark 延迟。

### Developer Trace API

- `POST /api/dev/session`：创建独立开发 session
- `POST /api/dev/turn`：只保存新 turn 输入
- `POST /api/dev/stage`：执行指定的下一个合法阶段
- `POST /api/dev/next`：只执行下一个阶段
- `POST /api/dev/all`：按顺序执行全部剩余阶段
- `POST /api/dev/restart`：明确重新开始当前 turn
- `POST /api/dev/commit`：提交完整 turn State
- `GET /api/dev/trace/{session_id}/{turn}`：读取已存历史 trace

## 当前限制

- Demo 与 Developer session 都只保存在服务器内存，服务器重启后清空。
- 现有 SQLite Retriever 有线程绑定限制，因此 Agent 与 Trace 操作由 Uvicorn event-loop 线程加锁执行。
- 本地执行时间包含前端适配器调用开销，不代表比赛官方延迟。
- Developer Mode 只展示结构化输入、输出、分数、排名和错误，不展示或伪造 chain-of-thought。
- Trace 参数默认只读，使用 evaluator 默认值：`top_k=10`、Retrieval `k=100`。
