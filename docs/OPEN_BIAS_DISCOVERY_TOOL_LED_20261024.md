# Open-ended Autonomous Bias Discovery — tool-led pilot (2026-10-24)

## Research intent and decision

The prior KuaiRand-27K study forced Qwen to select a candidate_id from 14 precomputed evidence cards and performed one model call. That measured candidate selection, not autonomous scientific question formation.

The new experimental entry is **explore**, separate from the existing immutable **review** record. The model now chooses repeated development-only SQL queries, creates conditional measures, submits a falsifiable concept, receives a separate Critic judgement and a literature check. It never gets direct holdout SQL access.

## Implemented scope

- `rsi_researcher/research/discovery/open_research.py`: development-only physical materialization, SQL tool, compact contextual memory, tool-led model loop, literature retriever + reviewer, numerical evidence journal.
- `rsi_researcher/research/discovery/__main__.py`: preserved status/prepare/review and added explore.
- `configs/discovery_qwen_open_20261024.json`: explicit reproducible vLLM sampling config, not DeepSeek.
- `tests/test_open_research.py`: separation and file-exfiltration controls, tool-led loop, literature retrieval failure, contradictory Critic, compaction.
- The legacy C00 outputs and full KuaiRand source files remain untouched.

### Data boundary

Source trajectory database:
`runs/kuairand27k-science-frozen-v1-20261024/seq.duckdb`

Separate development-only database:
`runs/kuairand27k-open-agent-development-20261024.duckdb`

This export contains **21,682,292** feature rows and **21,824,051** event rows. It includes only development-fold user IDs and records dated no later than April 30, 2022. It does not export `user_fold`, which would be a holdout identifier, nor the precomputed all-user cross-user feedback aggregate feature, which could indirectly incorporate confirmation users.

The agent's DuckDB handle opens only the development file, with read-only mode and external file access disabled. SQL input is restricted to a single SELECT/WITH; output is capped. Original complete SQL evidence is journaled, while prompt memory contains compact excerpts. This is a scoped research tool, not a general-purpose hostile-code execution sandbox.

### Active hypothesis formation

The model receives the current exact data schema on every step, not pre-ranked C00–C13 candidates. It can generate CASE/aggregation expressions, interactions, changed time windows and outcome measures from the development tables.

The Critic can request another development query, reject, or provisionally freeze. A proposal must contain an executable development measurement SQL. Programmatic freeze eligibility additionally requires:
- two prior successful exploratory queries (pilot setting);
- successfully executed proposed measurement query;
- Critic's `observable_valid` and `rival_discriminated` both true;
- a separate `freeze_candidate` decision.

This still **does not** prove an algorithmic bias, causal identification, scientific novelty, independent confirmation, or debias effectiveness. Machine-readable outputs explicitly indicate those checks remain false.

### Literature Researcher

A model generates mechanism-focused queries, OpenAlex supplies works (IDs, DOI, publication years and bounded abstract excerpts), a second model call compares novelty overlap. Returned paper IDs must come from retrieved hits. Network failure/incomplete retrieval forces `insufficient_literature`. Retrieval is a provisional novelty assessment rather than a systematic review of full texts.

### Reproduction

Run in `/data/cjh/rsi/project`:

```bash
PYTHONDONTWRITEBYTECODE=1 /data/cjh/rsi/discovery_private_20261024/env/bin/python \
  -m pytest -q tests/test_open_research.py

# With a Qwen vLLM instance running at the port in the config:
PYTHONDONTWRITEBYTECODE=1 /data/cjh/rsi/discovery_private_20261024/env/bin/python -B \
  -m rsi_researcher.research.discovery explore \
  --source-db runs/kuairand27k-science-frozen-v1-20261024/seq.duckdb \
  --development-db runs/kuairand27k-open-agent-development-20261024.duckdb \
  --output runs/REPLACE_WITH_NEW_UNIQUE_RUN \
  --config configs/discovery_qwen_open_20261024.json \
  --max-rounds 9 --min-queries 2
```

Every run records `development_manifest.json`, `journal.json`, `model_calls/`; completed runs save `frozen_candidate.json`.

## Real pilot log (do not conflate capability with validated discovery)

- v1: model made a real query then hallucinated nonexistent SQL columns, so no candidate was frozen.
- v2: three valid SQL queries, real OpenAlex results, a candidate was frozen provisionally. Post-review found Critic's prose denied evidential validity while its decision said freeze. This motivated explicit booleans and executable-measurement gating.
- v3: ran more dynamic queries, but accumulated prompt history exhausted the model's 16k context at the Critic step. Changed model-visible memory to compact summaries while keeping the full journal.
- v4: the compact evidence memory completed without a context error. Qwen executed six valid queries but its questions remained mostly simple assignment/previous-response correlations, and it did **not** meet the machine-checked freeze gate. Result: `runs/kuairand27k-open-agent-qwen-pilot-v4-20261024/frozen_candidate.json`, `status=no_frozen_candidate`. This is an important honest limit: tool access is present but strong concept formation has not yet been demonstrated.
- Focused independent literature search for the GPT-assisted mechanism retrieved 12 papers including `Recommender systems, ground truth, and preference pollution` (2022) and `Estimating and Penalizing Induced Preference Shifts in Recommender Systems` (2022), further weakening a simplistic novelty claim. These bibliographic results are in `runs/kuairand27k-open-agent-gpt-assisted-20261024/relevant_literature.json`.
- GPT-assisted exploration: manually authored three new conditional SQL analyses of elapsed-time bins and previous-click responses, including activity strata and equal-weight within-user contrast. Evidence is in `runs/kuairand27k-open-agent-gpt-assisted-20261024/development_probes.json`. This is **not** a blinded GPT API run or a fair GPT-Qwen comparison.

## Remaining research milestones

1. Blind, identical-tools GPT API/Qwen research trial, with equal budgets and independent new data/evaluation. Not established by the current manual GPT probe.
2. Separate and frozen test of unsupported concepts: user-disjoint/time-disjoint holdout, correct unit of inference, comparisons with baselines, multiple-testing accounting and credible utility reference. The new explore command intentionally **does not call this evaluator**.
3. If independently supported and mechanism-identifiable, intervention in a reproducible recommender, plus remove/restore and utility-sensitive Debias.
4. RSI cross-cycle Director with research-memory selection and cost-efficiency comparisons; this pilot provides the journal but is not a complete continuous autonomous RSI loop.

Do not promote a descriptive association or a provisional new term to a verified new Bias.

## 2026-10-24 v2：真正开放的 Concept–Experiment–Reflection 循环

**主线仍为** `python -m rsi_researcher.research.discovery explore`。这是对同一 Discovery 入口及 `open_research.py` 的演进，不是另起一个科研框架。旧 `review` 命令保留为历史 C00 实验复核，不再作为新的开放 Discovery 入口。

### P0 — Data Explorer

- 新增动作：`inspect_trajectory`、`define_feature`，并保留模型自行编写只读开发集 SQL。
- `define_feature` 显式指定 `source_table=events|features`；生成的是进程内临时视图，不改变官方日志；后续 SQL 可以继续写原始的 `FROM events` / `FROM features`，工具负责透明映射。
- 文本里提到的派生特征不自动等于真实存在。定义需先通过真实 SQL 计算；使用不存在字段的查询会报告错误。相同 SQL 重复执行不会获得新的科研证据。
- 所有原始结果保留到 `journal.json`，模型上下文只携带压缩片段；开发集与确认集仍物理隔离。

### P1 — Literature Researcher

- Agent 根据机制和竞争解释生成 2–3 个英文检索问题，向 OpenAlex 查询论文标题、年份、DOI、论文 ID 与摘要。只有检索返回的论文 ID 能被后续文献审查引用。
- 检索语句自动保持推荐系统领域语境，明显无关的同形词结果（例如 dolphin click、virus click）从审查证据中排除，但在 `excluded_irrelevant` 中保留。
- 文献结论为 `possibly_distinct` / `likely_known` / `insufficient_literature`，仅仅检索不到文章不能证明新颖。
- Qwen 针对 H001 的真实文献审查记录：
  `runs/kuairand27k-open-loop-v2-literature-reviewed-20261024/literature_review.json`。
  本轮检索留下 11 篇相关/邻近工作，模型结论为 `insufficient_literature`。这不是文献新颖性的证明。

### P2 — Active Experimentation、Critic、Reflection

- Data Explorer 开放动作：`sql`、`inspect_trajectory`、`define_feature`、`form_concept`、`test`、`revise`、`literature`、`freeze`、`stop`。模型自主决定工具调用及研究条件。
- 若多个初期工具调用之后尚未形成概念，Concept Formation Agent 根据已经测得的开发数据自主提出 2–3 个互相竞争的概念，附机制、反解释、反证条件、测量与候选 SQL，编号 H001/H002 等；编号只作研究标识，不是固定 Bias 卡片。
- 研究者选择 H001/H002 后，实际运行其 SQL；Critic 可以提出进一步 SQL 并实际执行；其结果回到科学家下轮上下文。实验日志保留原始假设、执行 SQL、测试版本及证据。
- 多次开发反证仍无法排除竞争机制时，调用 Reflection Agent 选择修订、放弃或查文献。修订保留历史版本与失败原因。
- **新科学质量审计**：只有单个结果组的查询不能证明对照效应；Critic 的标志被数值审计否决。Reflection 没有真正改变概念、机制、反证或测量时，不得冒充新版本。
- Freeze 是独立确认的候选提案，而不是 Bias 已证实。必须有真实开发实验、Critic 科学门槛和文献审查。所有输出仍写明 `independent_confirmation=false`、`causal_bias_established=false`、`debias_verified=false`。

### 真实 Pilot 证据与验收状态

- `runs/kuairand27k-open-loop-v2b-qwen-20261024/`：Qwen 自主形成 H001/H002，执行多轮假设测试和 Critic 反证，但没有形成可靠的新 Bias。
- `runs/kuairand27k-open-loop-v2e-qwen-20261024/`：实际完成了「形成假设 → SQL 检验 → Critic 补充实验 → Reflection 提出修订」调用链。但模型的修订基本复述原概念，不满足有意义的新测量改进。
- `runs/kuairand27k-open-loop-v2f-audited-20261024/`：对不实质改变的修订进行 veto，防止错误地把同一假设算成新版本。最终 `development_inconclusive`，没有 Holdout 读取。
- 目标科学能力比以前更接近：LLM 已能自主创造字段、提出不在固定卡片中的概念并驱动真实实验，但**有效机制形成与科学修订质量仍是 Qwen3-8B 的瓶颈**。不能把代码流程通过等同于新 Bias 得到了科学确认。
- 针对性测试与相邻回归测试：`tests/test_open_research.py`、`tests/test_discovery.py`、`tests/test_kuairand_temporal_discovery.py`，共 **23 passed**。

### 可复现入口

```bash
cd /data/cjh/rsi/project
PYTHONDONTWRITEBYTECODE=1 /data/cjh/rsi/discovery_private_20261024/env/bin/python \
  -m pytest -q tests/test_open_research.py tests/test_discovery.py tests/test_kuairand_temporal_discovery.py

# 需要独立启动并自主管理的本地 Qwen 服务；每次使用新的输出目录
PYTHONDONTWRITEBYTECODE=1 /data/cjh/rsi/discovery_private_20261024/env/bin/python -B \
  -m rsi_researcher.research.discovery explore \
  --source-db runs/kuairand27k-science-frozen-v1-20261024/seq.duckdb \
  --development-db runs/kuairand27k-open-agent-development-20261024.duckdb \
  --config configs/discovery_qwen_open_20261024.json \
  --output runs/NEW_UNIQUE_OPEN_DISCOVERY_RUN \
  --max-rounds 14 --min-queries 2
```

**下一阶段**：先用独立未见过的数据和一致工具预算正式对照 GPT 与 Qwen，比较有效的新变量、可区分竞争机制的实验和独立成功率，而不是只比较模型文字。现有 `explore` 不自动访问 Holdout 或执行 Debias；只有冻结且经独立证据支持的概念才进入下一层机制级干预评价。
