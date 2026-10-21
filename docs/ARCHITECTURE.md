# RSI 代码结构：唯一正式主线

**正式路径：** `/data/cjh/rsi/project`。你无需再进入带日期的 worktree 寻找最新实现。
**入口：** `python -m rsi_researcher llmrank-cycle`（持续多轮研究），或 `python -m rsi_researcher science-tree --task llmrank_bias`（单批次）。

## 项目导航

| 你要看什么 | 文件 | 职责 |
|---|---|---|
| 怎么运行 | `README.md` / `rsi_researcher/cli.py` | 统一命令入口 |
| **跨轮研究循环** | `rsi_researcher/llmrank/cycle.py` | 预算、Patience、逐轮证据继承、只在最后打开独立测试集 |
| **树搜索及数值审计** | `rsi_researcher/llmrank/science.py` | 当前候选、frontier、去重剪枝、准入、评估、Reflection、研究记忆 |
| Agent 定义 | `rsi_researcher/llmrank/agents.py` | Designer / Critic 专用 Schema，复用原 Diagnosis / Reflection |
| 原科研 Harness | `rsi_researcher/research/scientific_agents.py` | 严格研究代理框架、证据权威 |
| 原树搜索入口 | `rsi_researcher/research/scientific_tree.py` | 保留 CLI 分发及历史任务兼容 |
| 持续记忆/技能 | `rsi_researcher/research/scientific_skills.py` | 审计来源、Candidate Skill 不可自我认证 |
| 实际推荐及策略 | `rsi_researcher/llmrank/strategy.py` | LLM 排序 Prompt、Bootstrapping、后处理、完整排名校验 |
| 自由公式 | `rsi_researcher/llmrank/formula.py` | 受限算术 DSL，不允许 arbitrary Python |
| 数据划分及指标 | `rsi_researcher/llmrank/dataset.py` | Leave-two-out/官方 Games，HR/NDCG/ARP/APLT/PopREO |
| 作者候选适配 | `rsi_researcher/llmrank/candidates.py` | ML-1M、Games Random/SASRec 预计算候选 |
| Amazon 数据 | `rsi_researcher/llmrank/amazon.py` | Beauty、Sports、CDs 2014 5-core 转换 |
| Agent 只读证据工具 | `rsi_researcher/llmrank/tools.py` | 候选流行度、效用—偏差配对差异 |
| 实验报告 | `rsi_researcher/llmrank/report.py` | 指标前后对比 |
| 本地模型与资源 | `rsi_researcher/runtime/models.py`, `runtime/limits.py` | Qwen vLLM、JSON Schema、请求/Token/时间额度 |
| 模型配置 | `configs/llmrank_qwen.json` | 显式端口与模型参数，需服务预先启动 |
| 研究约束 | `docs/LLMRANK_RESEARCH_PROGRAM.md` | 冻结评价器、研究结论权威、Autoresearch 的可编辑表面 |
| 回归测试 | `tests/test_llmrank_mainline.py`, `tests/test_llmrank_cycle.py` | 隔离测试集、合法策略、跨轮停止、同一候选协议 |
| 数据目录 | `datasets/llmrank/`, `datasets/amazon/raw/` | 数据本体，不纳入 Git |
| 实验目录 | `runs/` | 每个实验不可覆盖，开发轮次/最终独立确认分开 |

## 当前核心研究工作流

```text
datasets/{llmrank,amazon}
  └─ Dataset+Candidate Adapter
       └─ frozen train / development users / unseen holdout
            ├─ Qwen LLMRank baseline + registered fixed debias
            │
            └─ Outer RSI Cycle (cycle.py)
                │  max rounds, global compute budget, patience
                └─ Evidence-guided tree (science.py)
                    ├─ Read-only diagnostics and independent baseline numbers
                    ├─ Diagnosis (scientific_agents.py)
                    ├─ Designer (llmrank/agents.py): generative strategy
                    ├─ Executability validator + Critic: protocol admissibility
                    ├─ Paired Qwen ranking, Qwen cache, bounded cost
                    ├─ Independent metric audit and Pareto tradeoff
                    ├─ Reflection, verified numeric Research Memory + Candidate Skill
                    └─ parent expansion, exact dedupe/pruning, backtracking
                 ↻ new development round uses all predecessor numeric audits
                    and the best inherited candidate (retested on frozen development)
            └─ Freeze best development candidate ONCE
               └─ independent_confirmation (cycle.py)
                  ├─ no more policy search or test-driven selection
                  ├─ baseline / selected fixed / selected RSI
                  ├─ paired user Bootstrap CI + raw Qwen ranking logs
                  └─ clear limits, including no causal or statistical claims
```

### 与上一版不同之处

- **不再建另一套顶层 RSI 项目**；统一 CLI、Runtime、Scientist 契约、数值审计、Skill provenance。
- 原始 `scientific_tree.py` 中 AgentCF/Coat 的执行契约不适用于 LLMRank，因此 LLMRank 专用任务在 `llmrank/science.py` 做了受控适配。仍有小部分任务特定调度逻辑，**尚不是完全消除重复代码**，后续优先整合共同的 tree frontier。
- `cycle.py` 可以跨轮，不再只运行一轮然后终止。过去的节点及明确失败指标传给新一轮，精确重复参数组合直接剪掉；不把低样本下偶然失败的整个策略族直接判为永远无效。
- **中间轮次禁止读取 holdout**，因此不能靠无限重试直到在同一测试集上出现好数字。最后一次确认之后结束本批科研。
- **一项公式只是一个算子**。当前策略也有历史选择、历史顺序、候选呈现、投票集成、动态 Prompt、后排流行度系数。更深层的去偏如检索重构、置信度条件去偏、个性化校准、曝光机制补偿、用户分群仍需实现，不冒称已完成。

### 为什么过去没找到稳定改进

观察到多轮「ARP/APLT 改善而 HR/NDCG 恶化」。这是有用的机制证据，不是有效最终去偏发现。潜在原因：

1. 流行度和用户相关性天然相关，统一惩罚热门会把真实用户兴趣一同抑制。
2. 只靠标题与有限用户历史，零样本 Qwen 很难判断长尾目标与当前用户是否真的匹配。
3. 候选集的召回偏差与 LLM 排序偏差会互相混淆；若候选集本身长尾少，靠排序难以修复。
4. 过去的研究样本仅 4–14 位用户开发面板，一个命中变化可引起离散的大波动；NDCG 硬门槛将误杀不少策略。
5. 旧 Critic 要求「足够因果反证」才允许试验，并以自然语言过早否决；现将准入与效果评价分离。
6. 策略空间缺乏**条件性去偏**：应对喜欢热门且确实相关的用户少惩罚，对过度集中而不伤效用的用户采用更强干预。
7. 缺少严格同 Token/调用成本的搜索算法比较；较高 ensemble 调用数并不等于科学效果更优。

### 后续优化的可检验优先级

1. 提高 development 用户数与分层覆盖，保留独立最终确认；引入长尾/热门用户分组、逐用户效用损失分布。
2. 在策略生成语言里加入**质量保护的 top-k 交换、条件性惩罚、个性化流行度校准、原始排序不确定性**，不只改一个线性公式。
3. 多保真搜索：小样本先筛不可执行与明显无效，大样本复查 Pareto Frontier；对已证实的强退化方向做软剪枝、避免重复。
4. 停止条件含 max_rounds、全局 requests/tokens、patience、候选耗尽。找到开发胜者仍需要唯一一次 untouched heldout 测试；失败不能继续用同一留出集调参。
5. 比较固定规则、随机搜索、Agent Tree 的**匹配实际成本**，看 RSI 是否得到更多可迁移发现，而不仅是某次降低 ARP。

## 本目录维护原则

这份文档及 `README.md` 指向唯一最新日常入口。历史 AgentCF/Coat 研究保存在 Git 版本记录和单独的归档目录；不要把旧实验的数据/评价器重新加入当前 LLMRank 工作流。清理要保留科学必需的运行时及可恢复证据，不为了表面目录数减少而破坏已有软件依赖。

## 新增的条件性去偏算子

`rerank_policy` 不再只有全体商品统一减去 `lambda * pop`：

- `global`：原始的所有用户统一流行度惩罚，作为机制对照。
- `personalized_gate`：先计算当前用户训练历史的流行度中位数，**只惩罚超过其熟悉基准的候选**。测试目标与留出反馈不参与门槛计算。
- `protect_top3`：从原 Qwen Borda 排序中固定前三位，剩余候选执行自由公式与流行度惩罚排序。明确限制高排位变动的代价。

这些都是可执行的机制，而不只是改变 Prompt 文案；它们不能自动保证 NDCG 提升，需与强规则在同样的冻结候选与计算预算下对照。

**跨轮剪枝目前谨慎：** 精确重复策略、非法公式及已经确定不可执行的提案会被排除；针对整个策略族的硬剪枝只有在多个独立开发子群上确认可复现的失败方向后才合理。12 用户的小样本不足以证明一个策略族永远无效。搜索器把表现为「偏差下降但相关性下降」的节点保存为待改进的 Pareto 候选，而不是直接从历史中丢弃。
