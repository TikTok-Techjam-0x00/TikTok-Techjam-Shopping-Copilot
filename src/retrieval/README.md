# 模块 1 — 检索（Retrieval）

当前生产基线：`bm25_v0`；Dense与Hybrid暂时作为可切换实验策略。

## 目录结构

```text
src/retrieval/
  catalog.py / text.py / query.py       数据、商品文本与查询构造
  bm25.py / dense.py / hybrid.py        可复用检索算法
  embedding.py / multivector.py         向量缓存与多向量实现
  retriever.py / __init__.py            给Pipeline和3A使用的稳定入口
  evaluation/
    first_turn.py                       首轮严格Recall评测
    multiturn.py                        独立多轮全流程Retrieval评测
  experiments/
    README.md / PLAN.md / EXPERIMENTS.md 实验说明、规划和登记表
    text_ablation.py                    BM25文本消融
    query_instruction.py                Dense Query对照
    dense_text.py                       Dense商品文本对照
    hybrid_comparison.py                BM25/Dense/Union/Fusion对照
    visualize_results.py                JSON转离线HTML报告
  tools/build_embeddings.py             离线生成商品embedding
  tests/                                Retrieval单元测试
```

根层只保留生产运行会复用的代码，实验、评测、工具和测试各自归类。详细实验执行
顺序见 `experiments/PLAN.md`，已完成与待运行方案见 `experiments/EXPERIMENTS.md`。

## 商品文本版本

商品文本由 `src/retrieval/text.py` 统一构造。BM25保留独立字段以应用列权重；
后续Dense可以复用同一版本生成带标签的统一字符串，embedding缓存必须记录
`text_version`。

当前版本：

```text
title_v0             title
title_category_v1    title + categories
core_v2              title + categories + features
core_attributes_v3   core + selected normalized attributes
all_fields_v4        title + categories + features + details + store + description
dense_attributes_v2  compact title + official 10-field-derived high-signal values
dense_attributes_v2_unlabeled  same values without repeated labels
dense_identity_v1    title + product type + brand
dense_needs_v1       material + color + size + style + use_case + features
```

默认仍为 `all_fields_v4`，与原 `bm25_v0` 的字段范围一致。运行全部文本消融：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.experiments.text_ablation
```

也可以在VSCode直接运行 `src/retrieval/experiments/text_ablation.py`。结果默认写入：

```text
artifacts/bm25_text_ablation.json
```

单独评估一个版本：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.evaluation.first_turn `
  --text-version core_v2 `
  --experiment bm25_core_v2
```

## Dense Retrieval与Embedding缓存

Dense Retrieval同样默认使用 `all_fields_v4`：

```text
50,000 Item
  -> build_product_text(all_fields_v4)
  -> text-embedding-v4（默认256维）
  -> L2归一化
  -> embeddings.npy
```

第一次运行前，复制 `.env.example` 为 `.env` 并配置有效的DashScope Key。
新加坡通用OpenAI兼容地址是：

```text
https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

如果使用Workspace专属地址，地址中的Workspace ID和API Key必须属于同一个
Workspace。批量生成并缓存全部商品embedding：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.tools.build_embeddings
```

也可以在VSCode直接运行 `src/retrieval/tools/build_embeddings.py`。默认缓存目录为：

```text
artifacts/retrieval/dense/
  text-embedding-v4__all_fields_v4__d256/
    manifest.json
    parent_asins.json
    embeddings.npy
```

`manifest.json` 会记录模型、维度、文本版本、商品数量和Catalog文本指纹；任一项
不一致都会拒绝加载。仓库通过 Git LFS 发布了已验证的三套完整50,000商品缓存：
`all_fields_v4`、`dense_attributes_v2` 与 `dense_attributes_v2_unlabeled`。拉取
最新代码后运行 `git lfs pull` 即可下载缓存，测试时无需重新调用商品embedding API。
在线查询仍需在 `.env` 中配置可用Key。批量任务每个batch记录进度，中断后再次运行
可以继续，不会重新编码已经完成的商品。构建脚本默认使用1个请求线程，以避免长文本
Catalog超过服务端每分钟Token限制；可通过
`--workers` 调整，但应先观察账号的TPM限制：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.tools.build_embeddings --workers 1
```

在线检索时只编码查询，然后用归一化查询向量和50,000商品矩阵做点积；因为两边
都经过L2归一化，所以点积就是余弦相似度。`dense_score` 与
`retrieval_score` 都是越高越好。

Dense查询支持三个明确分开的实验模式：

```text
symmetric          OpenAI-compatible默认编码，作为原始基线
query              DashScope text_type=query，不加任务指令
query_instruction  DashScope text_type=query，并加入英文检索任务指令
```

`text_type` 和 `instruct` 是DashScope原生接口能力，不由OpenAI-compatible接口
提供。实现仅将原生接口用于在线查询；现有50,000个document embedding缓存可以
原样复用。运行对照实验：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.experiments.query_instruction
```

结果默认保存到 `artifacts/dense_query_instruction_v1.json`。

## BM25 + Dense Hybrid

Hybrid首先分别取得BM25和Dense候选：

```text
BM25 Top-K ----\
                -> Candidate Union -> RRF / Weighted Fusion -> strict Top-K
Dense Top-K ---/
```

`Candidate Union` 使用精确 `parent_asin` 去重，最多包含 `2K` 个商品。它表示
候选池召回上限，没有可直接比较的跨检索器分数。RRF只使用两个检索器的名次：

```text
score = 1 / (60 + bm25_rank) + 1 / (60 + dense_rank)
```

Weighted Fusion先分别对BM25分数和Dense余弦分数做min-max归一化，再计算：

```text
score = alpha * normalized_bm25 + (1 - alpha) * normalized_dense
```

默认 `alpha=0.5`，应根据公开集实验调整，不能直接相加原始BM25分数和余弦分数。
如果在线Dense查询调用失败，`HybridRetriever` 默认回退到BM25；实验脚本则应在
有效缓存和API配置下运行，避免把fallback误记为Dense结果。

一键比较BM25、Dense、Union、RRF和Weighted Fusion：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.experiments.hybrid_comparison
```

默认结果保存到：

```text
artifacts/retrieval_hybrid_comparison.json
```

其中Union的 `Recall@K` 定义为目标进入“BM25 Top-K或Dense Top-K”，对应物理
候选池最多为 `2K`；RRF和Weighted Fusion的 `Recall@K` 是严格排序后的Top-K。

## 数据流程

```text
catalog.jsonl(.gz) -> Catalog[parent_asin, Item] -> SQLite FTS5
当前查询 + ShoppingState -> 检索查询 -> BM25 -> list[Candidate]
```

每个 `Item` 还提供统一的 `item.attributes` 派生属性，供 Retrieval 后续
metadata 实验、3A 约束匹配和 3B 候选属性分析共同使用。属性在第一次访问时
才提取并缓存，因此加载 50,000 个商品和构建 BM25 时不会提前承担全部抽取开销。
`Item.to_dict()` 仍只返回官方 Catalog 字段。

生产代码统一使用官方10字段属性契约：

```text
category, material, color, size, style,
brand, budget, feature, use_case, other
```

旧详细版本中的 `fit/pattern/target_user/quantity` 不再由 Retrieval 作为独立字段
消费；它们由最新 `attribute.py` 归并到上述10字段中的 `style/feature/other`。

如需把全部商品属性导出为独立 JSONL，可在项目根目录直接运行：

```powershell
.\.venv\Scripts\python.exe extract_attributes.py
```

默认输出 `data/catalog_attributes.jsonl`，该生成文件已被 Git 忽略。

模块的公共调用入口如下：

```python
from src.retrieval import Catalog, BM25Retriever

catalog = Catalog.load("data/catalog.jsonl")
retriever = BM25Retriever(catalog)
candidates_100 = retriever.retrieve(
    query=user_message,
    state=shopping_state,
    intent=shopping_state.intent,
    k=100,
)
```

## 输出约束

- 返回 `src/item.py` 中定义的完整共享 `Candidate` 对象；
- `parent_asin` 必须来自冻结 Catalog，并保持精确、唯一；
- 候选按照从优到劣排列，`retrieval_rank` 从 1 开始；
- `bm25_score` 和 `retrieval_score` 都遵循**分数越高越好**；
- 空的文本查询返回空列表；
- `k` 可配置，不强制只能返回 100 个候选。

SQLite 原生 BM25 分数是越低越好，因此本实现会先对原生分数取负，再将其作为对外分数。标题和类别的字段权重最高，与官方 starter 的基线设计保持一致。

Catalog Loader 会跳过格式错误的记录；如果出现重复 ASIN，则保留第一个有效商品，并通过 `catalog.stats` 提供加载诊断信息。

## ShoppingState 查询构造

如果 ShoppingState 中存在当前有效的类别或约束，查询构造器会把当前状态视为权威信息来源，而不是盲目拼接全部历史消息。

这样可以避免 Module 2 完成 intent override 后，旧偏好再次进入检索查询。例如用户从 `running shoes` 改为 `hiking boots`，只要 Module 2 已删除旧约束，Retrieval 就只会使用当前状态中的 `hiking boots`。

目前纯数字范围约束不会转换成 BM25 关键词。例如预算上限 `$100` 会继续保存在 ShoppingState 中，交给后续 metadata 实验或 3A 使用，避免无意义的数字给 BM25 带来噪声。

## 运行测试

使用项目虚拟环境运行全部测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

也可以在 VSCode 中直接打开并运行：

```text
src/retrieval/tests/test_retrieval.py
src/retrieval/tests/test_evaluation.py
```

`test_retrieval.py` 验证 Catalog、查询构造、BM25、Top-K、去重、分数方向和确定性；
`test_evaluation.py` 验证首轮/多轮Recall、排名、场景拆分和累计命中是否正确。

## 运行 Recall 评估

执行可重复的首轮 Recall 评估：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.evaluation.first_turn `
  --catalog data/catalog.jsonl `
  --dataset data/public_set.jsonl `
  --ks 10 50 100 `
  --output artifacts/retrieval_bm25_v0.json `
  --include-sessions
```

也可以在 VSCode 中直接运行：

```text
src/retrieval/evaluation/first_turn.py
```

评估程序复用官方 public simulator，为每个 session 生成首轮用户消息。隐藏目标 ASIN 只用于生成官方模拟消息和判断返回排名，绝不会作为输入传给 Retriever。

这个命令测量的是 Retrieval 的首轮 Recall，不是官方完整多轮评估中的 Hit Rate、MRR 或 MTTC。特别是 Intent Override 场景，在 override 正式发生前出现目标并不会被官方完整 evaluator 计为命中，因此首轮 Recall 只能作为检索诊断指标。

## 独立多轮全流程评测

多轮评测复用官方用户模拟、Module 2状态累计和当前3B追问策略，但在进入3A之前直接
记录 Retrieval Top-K。默认严格遵循比赛停止规则：普通场景目标首次进入Top10即停止；
Intent Override只有在override生效后的Top10命中才会停止：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.evaluation.multiturn `
  --method bm25 `
  --ks 10 50 100 `
  --max-turns 10 `
  --output artifacts/retrieval/multiturn_bm25_v1.json
```

可选方法为 `bm25`、`dense`、`hybrid_rrf` 和 `hybrid_weighted`。结果同时包含：

- 每轮严格 `Recall@10/50/100`；
- 截至每轮的累计 `Session HitRate@K`；
- Buying、Browsing、Boundary、Intent Override场景拆分；
- 每个Session每轮的message、state query、目标rank、延迟和Top-N结果；
- Intent Override发生前的轮次标记为不可计分，不混入严格Recall分母。

如果需要观察命中后的反事实查询变化，可额外传入 `--continue-after-hit`。这些后续轮次
会标记为 `post_hit_counterfactual=true`，并只进入 `diagnostic_recall`，不进入官方
严格逐轮Recall分母。

生成可视化HTML：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.experiments.visualize_results `
  artifacts/retrieval/multiturn_bm25_v1.json
```

该评测用于定位 Retrieval 问题，不替代官方端到端Evaluator；3B会影响下一轮用户披露
什么，但目标标签不会传入Retrieval或3B。

## 已测基线

`bm25_v0` 于 2026-08-27 在全部 50,000 个 Catalog 商品和 200 个公开 session 上完成测量。查询使用未修改的官方 evaluator 生成的首轮消息。

| 版本 | Recall@10 | Recall@50 | Recall@100 | 平均查询延迟 | P95 查询延迟 |
|---|---:|---:|---:|---:|---:|
| `bm25_v0` | 0.185 | 0.380 | 0.525 | 11.7 ms | 27.8 ms |

对应的目标命中数量为：

```text
Top 10：37 / 200
Top 50：76 / 200
Top 100：105 / 200
```

在当时的测量机器上，Catalog 解析耗时 0.681 秒，内存 FTS5 索引构建耗时 1.938 秒。延迟与机器环境有关，主要实验比较指标仍然是 Recall。这些结果只是 Retrieval 的首轮指标，不代表完整流水线的 Hit Rate、MRR 或 MTTC。

## 商品文本消融结果

`bm25_text_ablation_v1` 于2026-08-28在相同50,000商品和200个公开session上运行：

| 文本版本 | Recall@10 | Recall@50 | Recall@100 | 索引构建 |
|---|---:|---:|---:|---:|
| `title_v0` | 0.085 | 0.205 | 0.265 | 0.332 s |
| `title_category_v1` | 0.160 | **0.405** | 0.515 | 0.579 s |
| `core_v2` | 0.170 | 0.315 | 0.480 | 1.540 s |
| `core_attributes_v3` | **0.190** | 0.355 | **0.525** | 15.760 s |
| `all_fields_v4` | 0.185 | 0.380 | **0.525** | 2.518 s |

结论：保持 `all_fields_v4` 为默认。`core_attributes_v3` 只提升1个Top10命中、
没有提升Top100，并显著增加全量属性索引构建时间。`title_category_v1` 的Browsing
Recall@100达到0.525，而 `all_fields_v4` 的Buying Recall@100达到0.5875，说明两者
具有互补性；下一步优先实验Candidate Union、RRF和Buying/Browsing路由，而不是
用单一文本版本替换当前默认。

## Dense与Hybrid实测结果

`hybrid_qwen_v1` 于2026-08-28使用以下配置完成全量公开集测试：

```text
Catalog：50,000
Public sessions：200
Product text：all_fields_v4
Embedding：text-embedding-v4，256维
BM25/Dense source K：100
RRF constant：60
Weighted alpha：0.5
```

| 方法 | Recall@10 | Recall@50 | Recall@100 |
|---|---:|---:|---:|
| BM25 | 0.185 | 0.380 | **0.525** |
| Dense | 0.160 | 0.290 | 0.385 |
| Candidate Union | **0.250** | **0.460** | **0.585** |
| RRF | 0.200 | 0.400 | 0.495 |
| Weighted (`alpha=0.5`) | 0.230 | 0.390 | 0.485 |

Candidate Union不是严格Top-K排序：Union Top100表示目标进入BM25 Top100或Dense
Top100，平均产生172.34个唯一候选，最多200个。它把Top100命中从105提升到117，
说明Dense确实补回了12个BM25遗漏目标；但当前等权融合在截断到严格Top100时又
丢失了部分强BM25候选。

Weighted Fusion的BM25权重消融：

| alpha | Recall@10 | Recall@50 | Recall@100 |
|---:|---:|---:|---:|
| 0.5 | **0.230** | 0.390 | 0.485 |
| 0.6 | 0.225 | 0.400 | 0.490 |
| 0.7 | 0.215 | **0.405** | 0.495 |
| 0.8 | 0.215 | **0.405** | 0.510 |
| 0.9 | 0.180 | 0.400 | **0.525** |

因此当前默认严格Top100仍应使用 `all_fields_v4` BM25。`alpha=0.5` 更偏向改善
Top10，`alpha=0.9` 的Top100只能追平BM25；Hybrid暂时保留为实验策略，不能宣称
已经全面优于基线。下一步应研究不超过100候选的预算分配、非对称融合和Dense
查询/文本构造，而不是直接修改3A去接收200个候选。

完整结果保存在Git忽略的：

```text
artifacts/retrieval_hybrid_comparison.json
```

Dense实验把200条查询先批量编码，总计约3.047秒；结果中的Dense查询延迟约
1.535ms只包含50,000×256矩阵检索，不包含外部API查询编码时间。

## Query instruction实验结果

`dense_query_instruction_v1` 于2026-08-29复用同一份 `all_fields_v4`、256维、
50,000商品embedding缓存，在全部200个公开session上完成。三个模式只改变查询
向量生成方式，商品向量、查询文本和Recall计算均保持不变：

| 查询向量模式 | Recall@10 | Recall@50 | Recall@100 | Top100命中 |
|---|---:|---:|---:|---:|
| `symmetric` | 0.160 | 0.290 | 0.385 | 77/200 |
| `query` | 0.160 | 0.290 | 0.385 | 77/200 |
| `query_instruction` | **0.180** | **0.360** | **0.440** | **88/200** |

任务指令为：

```text
Given the current shopping request, retrieve catalog products that best match
the requested product type and disclosed constraints. Prioritize the product
type and hard requirements; treat soft preferences as secondary signals.
```

相比原Dense基线，instruction在Top10/50/100分别增加4、14、11个目标命中。
其中Browsing Recall@100由0.275升至0.3375，Intent Override由0.5667升至0.7000。
单独设置 `text_type=query` 的排名没有变化；向量级检查显示它与symmetric并非缓存
误用，而是几乎相同（示例余弦相似度0.9999992）。加入instruction后示例余弦降至
0.9517并带来可测量Recall提升。因此后续Dense/Hybrid实验应默认使用
`query_instruction`，但BM25默认策略不受影响。

把该查询模式接入原 `all_fields_v4` Hybrid 后，实测结果如下：

| 方法 | Recall@10 | Recall@50 | Recall@100 |
|---|---:|---:|---:|
| BM25 | 0.185 | 0.380 | 0.525 |
| Dense + instruction | 0.180 | 0.360 | 0.440 |
| Candidate Union | **0.265** | **0.505** | **0.630** |
| RRF | 0.225 | 0.390 | 0.525 |
| Weighted (`alpha=0.5`) | 0.235 | 0.405 | 0.530 |

相比未加instruction的Hybrid，Union Top100由117增至126个命中。严格Top100中，
RRF由0.495升至0.525，Weighted由0.485升至0.530。Weighted只比BM25多命中1个，
因此仍需下游完整评估后才能替换默认策略。完整结果位于Git忽略的
`artifacts/retrieval_hybrid_query_instruction.json`。

## dense_attributes_v2与多向量表示

`dense_attributes_v2` 只消费生产版官方10字段contract，不引用旧详细schema。
它不会机械拼入所有字段：`budget` 保留给metadata比较，`other` 默认排除；Size只
使用标签尺码，不加入package/product dimensions；Features最多保留1200字符，避免
长字段淹没商品类型和硬属性。

为回答重复标签是否影响准确率，代码同时注册两个内容完全对应的版本：

```text
dense_attributes_v2            Title: ... / Product type: ... / Material: ...
dense_attributes_v2_unlabeled  ...（相同值，但移除字段标签）
```

必须分别生成全量缓存并对照Recall，不能只凭直觉选择：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.tools.build_embeddings --text-version dense_attributes_v2
.\.venv\Scripts\python.exe -m src.retrieval.tools.build_embeddings --text-version dense_attributes_v2_unlabeled
```

严格对照已于2026-08-29完成。三套缓存共享同一批173个唯一Query instruction
向量，以下均为最终列表不超过K的严格Recall：

| 商品文本 | Recall@10 | Recall@50 | Recall@100 | Top100命中 |
|---|---:|---:|---:|---:|
| `all_fields_v4` | **0.180** | **0.360** | 0.440 | 88/200 |
| `dense_attributes_v2` | 0.135 | 0.335 | **0.475** | **95/200** |
| `dense_attributes_v2_unlabeled` | 0.130 | 0.300 | 0.400 | 80/200 |

有标签版本在Top10/50/100分别比无标签版本多命中1、7、15个目标，说明短字段标签
没有因跨商品重复而降低效果，反而帮助模型识别字段语义。因此选择
`dense_attributes_v2`，无标签版本不进入后续实验。相对 `all_fields_v4`，紧凑
attributes文本在Top100多命中7个目标，但Top10少9个、Top50少5个：它扩大了
后段召回，却不适合作为当前头部排序的单独替代品。完整结果位于：

```text
artifacts/dense_product_text_query_instruction_v1.json
```

保持BM25使用 `all_fields_v4`、只把Dense切换为 `dense_attributes_v2` 后：

| 方法 | Recall@10 | Recall@50 | Recall@100 | 口径 |
|---|---:|---:|---:|---|
| Candidate Union | 0.240 | 0.500 | 0.625 | 非严格，Top100平均168.63个候选 |
| RRF | 0.195 | **0.430** | **0.535** | 严格Top-K |
| Weighted (`alpha=0.5`) | 0.180 | 0.400 | **0.535** | 严格Top-K |

虽然attributes Dense自身Top100更高，但Union比 `all_fields_v4` Dense组合的0.630少
1个目标，说明新增命中与BM25重叠较多。严格RRF/Weighted Top100提升至107/200，
但Top10下降。因此它可以作为多路召回或Buying策略的候选，不应直接替换当前默认。
完整Hybrid结果位于：

```text
artifacts/retrieval_hybrid_dense_attributes_v2.json
```

多向量表示将商品拆成两个独立缓存：

```text
identity = Title + Product type + Brand
needs    = Material + Color + Size + Style + Use case + Features
```

`MultiVectorDenseRetriever` 用同一个Query instruction向量分别计算两组余弦相似度，
支持加权融合与max融合。加权融合要求商品身份和需求属性共同匹配；max融合更偏召回，
允许任一路强匹配进入候选。对应缓存命令：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.tools.build_embeddings --text-version dense_identity_v1
.\.venv\Scripts\python.exe -m src.retrieval.tools.build_embeddings --text-version dense_needs_v1
```

## 后续实验

在更改默认策略前，建议依次进行以下可测量实验：

1. 生成identity/needs双向量缓存，并比较max与加权相似度；
2. 对当前最佳首轮策略运行完整多轮Evaluator并记录逐轮严格/累计Recall；
3. 调整混合文本Hybrid的RRF常量和Weighted Fusion alpha；
4. 分析Hybrid仍未进入Top-100的目标商品；
5. 实验Buying/Browsing路由和metadata boosting；
6. 重点记录Recall@10、Recall@50、Recall@100、延迟和缓存大小。
