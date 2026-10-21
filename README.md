# RSI: LLMRank + autonomous debias, 2026-10-21 mainline migration

**Active research branch**: feat/rsi-integrated-llmrank-20261021
**Development checkout**: /data/cjh/rsi/integrated-llmrank-20261021

The active recommendation-debias task is **llmrank_bias**. The old
AgentCF/Coat/KuaiRand evidence and code below are preserved for
history and regression; they are NOT the subject or dataset of
the new experiments. Main was not modified.

The original CLI, vLLM runtime with independent budgets, schema-governed
Diagnosis/Designer/Critic/Reflection, scientific skill provenance and
independent numeric evidence authority are retained. The LLMRank
dataset/candidate/reranking/scientific task adapters live in
rsi_researcher/llmrank. These are not a second top-level RSI app.

Run from this checkout with an *already running, separately authorized*
Qwen service:

    /data/cjh/rsi/envs/research/bin/python -B -m rsi_researcher science-tree --task llmrank_bias --config configs/llmrank_qwen.json --data /data/cjh/rsi/llmrank-rsi-20261021/external --dataset ml-1m --retrieval random --cohort-size 24 --seeds 20261021,20261022,20261023 --max-nodes 2 --max-requests 600 --output runs/UNIQUE_NAME

To run Games change --dataset Games. The Amazon Beauty, Sports, CDs
converter is in rsi_researcher/llmrank/amazon.py, but original Amazon
2014 5-core sources were not yet fully downloaded/verified, so no
valid result on those three categories is claimed.

The experiment contract, editable-vs-immutable boundaries and literature
inspiration are in docs/LLMRANK_RESEARCH_PROGRAM.md. Observe numeric
results with python -m rsi_researcher.llmrank.report runs/.../seed-...
Check independent result.json, frozen_evaluator_only.json,
model calls.jsonl, per-user rank traces, tree_index.json,
candidate_skill.json, and the scientific_memory.json.

## Measured early evidence (real Qwen, not a confirmed discovery)

ML-1M, 12 users (7 development, 5 holdout):
- Development baseline NDCG@5 0.27835, ARP@5 491.09, APLT@5 0.65714.
- Agent N001 NDCG@5 0.21429, ARP@5 354.67, APLT@5 0.83333.
- Bias proxy improves but relevance degrades, so N001 is not deployed.

ML-1M, 24 users (14 development, 10 holdout):
- Development baseline NDCG@5 0.12090, ARP@5 420.17, APLT@5 0.64286.
- Fixed recency NDCG@5 0.20933, ARP@5 372.71, APLT@5 0.68571, passes development.
- Agent N001: (NDCG@5 0.03076, ARP@5 342.09, APLT@5 0.74286).
- Agent N002: (NDCG@5 0.07583, ARP@5 257.30, APLT@5 0.78000).
- On the heldout group, baseline NDCG@5 0.21309, ARP@5 525.58;
  fixed recency NDCG@5 0.06309, ARP@5 404.18.
- Both Agent branches are rejected on development utility guard.
  Development success of a fixed rule did NOT generalize to holdout.

The current search combines seven strategy dimensions, including
a user-generated safe score expression. No arbitrary Python code or
edits to the evaluator/test set are allowed. A critic approves **protocol
admissibility**, never self-certifies debias effectiveness.
Small samples are exploratory. Real compute costs differ between fixed
and agent arms and must not be called strict equal-budget comparison.

---

# Prior RSI scientific architecture (historical research track)

# RSI Researcher：推荐系统偏差的可证伪多智能体科研系统

> 当前主线：`/data/cjh/rsi/project` · 分支 `main`。2026-10-20 已将 `feat/rsi-multiagent-20261017` 快进整合至主线。

## 科学目标

将 Selection、Exposure、Popularity、Filter Bubble、Fairness 五类经典推荐偏差转化为具有**可识别前提、合法实验、预设反证、独立评价与停止条件**的研究任务。Qwen3-8B 可提出假设与方案，但**不得修改实验执行器、评估代码、数据划分或科学通过标准**。

核心研究问题不是“模型能否给出更高 NDCG”，而是：**在未见任务和匹配成本下，证据驱动的多智能体研究能否比固定规则和强去偏算法找到更多独立确认的有效去偏发现？**

## 当前真正运行的架构

```text
Bias Task Registry / 固定数据与评价协议
    │
Outer Research Director（任务组合、全局预算、停止条件）
    │
Evidence-Guided Scientific Tree（假设分叉、frontier、父节点、局部预算）
    │
Diagnosis → Designer → Critic → 受控 Python Executor → Independent Numeric Evaluator → Reflection
    │                               │                                  │
    └── Verified Development Audit / Research Memory ←────────────────┘
                    │
          Candidate Research Skill（未确认）
                    │
      预注册未见数据 / 独立统计确认 / 外部批准后才可升级
```

- 五类 Bias 的概念定义见 `research/bias_registry.py` 与 `configs/bias_tasks.json`。
- 四个科研 Agent 的职责与结构化调用见 `research/scientific_agents.py`。
- 单回合科学协议在 `research/scientific_loop.py`；树搜索在 `scientific_tree.py`；全局任务与预算调度在 `scientific_director.py`。
- 注册干预、冻结 AgentCF 与独立实验测量在 `scientific_experiments.py`；五类开发数据在 `five_bias_development.py`。
- 策略对照在 `scientific_comparison.py`；候选科研技能在 `scientific_skills.py`。
- `recommenders/agentcf.py` 的 User/Item Memory 是研究**对象与干预机制**；旧 `research/memory.py` 属于历史科研入口，当前权威科研记忆使用来源可验证的独立数值审计，二者不可混淆。

## 运行（已有开发数据，无需启动 GPU）

```bash
cd /data/cjh/rsi/project
/data/cjh/rsi/envs/research/bin/python -B -m rsi_researcher bias-tasks
/data/cjh/rsi/envs/research/bin/python -B -m rsi_researcher bias-five --output runs/UNIQUE_FIVE_BIAS_RUN
/data/cjh/rsi/envs/research/bin/python -B -m rsi_researcher science-debias --help
/data/cjh/rsi/envs/research/bin/python -B -m rsi_researcher science-tree --help
/data/cjh/rsi/envs/research/bin/python -B -m unittest discover -s tests -t . -q
```

真实 Qwen 研究需已有获授权的本地 vLLM 服务、相应配置与独立计算预算；**上述命令不会擅自启动 GPU 服务**。所有运行输出必须写入新的、互不覆盖的目录；切勿用同一 Coat 保留集重新挑选候选。

## 可复核证据与科研结论

| 主题 | 开发阶段的发现 | 科学边界 |
|---|---|---|
| Coat Selection | cap=100 开发 Brier 改善 0.003735；冻结保留集改善 0.0034348 | 保留 NDCG 比例仅 0.980697，未达到 0.99，联合验收失败 |
| KuaiRand Exposure | 等量随机曝光训练 Brier 0.145696 vs 标准日志 0.226810（共同支持集） | 是随机参考分布上的预测证据；真实 logging propensity 不完整，非部署因果 OPE |
| Coat Popularity | \u03bb=0.10 时头部曝光 0.21931→0.21379 | 已知相关候选机会参照不支持直接宣称“纠正了不合理热门偏差” |
| Filter Bubble | 配对受控模拟熵 +0.174780；期望点击率 -0.017689 | 真实 KuaiRand 两期结果无法识别匹配的多轮因果干预 |
| Coat Fairness | Brier 差距 0.005924→0.004623；总体 Brier 0.149546→0.148983 | 仅开发集的二元群体预测误差 parity 候选，未独立确认 |
| Qwen 科学搜索 | 四角色、受控实验、独立审计、Outer Director / Tree 均已实际运行 | 固定规则在当前组合任务的开发门槛和调用成本上优于 Qwen；无 RSI 优越性证明 |

**独立确认的有效去偏发现数量：0。** 不把模拟机制有效、局部代理改善、程序测试通过或研究 Agent 语言表述算作独立科研发现。

不可覆盖的来源：`runs/coat-selection-preregistered-holdout-20261020-01`、`runs/five-bias-unified-development-20261020-04-normalized-dr`、`runs/five-bias-{fixed,qwen}-two-goals-20261020-01`。最新独立工作树的相关原始报告已按相同文件名复制到主工程的 `runs/` 以供检查；既有目录只读、不得复用为新确认。

## 文档

- [现行整体架构](docs/ARCHITECTURE_CURRENT.md)
- [下一阶段科研路线](docs/RESEARCH_ROADMAP_CURRENT.md)
- [真实五类开发与冻结确认结果](docs/FIVE_BIAS_CONFIRMED_RESEARCH_20261020.md)
- [本地 Qwen 与树/线性对照](docs/SCIENTIFIC_REAL_QWEN_RUN_20261020.md)
- [论文与源代码参照](docs/SCIENTIFIC_RESEARCH_REFERENCES.md)

`docs/` 中其余日期化记录及旧 `demo/run/study` 等 CLI 仍用于**历史实验重现**，不属于当前推荐的科研开发路径。不应误以为历史 ExperimentJournal/ResearchMemory 或旧 AgentCF 机制任务是新增的正式 Bias 类型。
