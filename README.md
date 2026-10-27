# RSI — 基于 LLMRank 的自主推荐去偏科研系统

**打开这个目录即可看最新代码：`/data/cjh/rsi/project`。** 不用再从带日期的 Git 工作树或 `experiments/` 中寻找实现。
**一张完整文件地图：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**；研究原则：[docs/LLMRANK_RESEARCH_PROGRAM.md](docs/LLMRANK_RESEARCH_PROGRAM.md)。

**按目标 Bias 判定与质量保护：[docs/BIAS_EVALUATION.md](docs/BIAS_EVALUATION.md)**（Popularity/LLM Candidate Position 两种新主线目标，分开评估）。

**最新自主策略程序设计：[docs/OPEN_POLICY_RESEARCH.md](docs/OPEN_POLICY_RESEARCH.md)**。Agent 现在可以构造非线性用户条件与多阶段重排，而不是只枚举系数。

## 最新主线

```
rsi_researcher/
├── cli.py                    # 单一命令入口
├── llmrank/
│   ├── cycle.py              # 外循环：跨轮研究、剪枝、预算、最终确认
│   ├── science.py            # 内循环：Agent → Critic → 执行 → 证据 → 树搜索
│   ├── agents.py             # LLMRank 任务级 Agent 协议
│   ├── strategy.py           # 可组合推荐/去偏实验机制
│   ├── formula.py            # 自主分段条件评分表达式
│   ├── program.py            # Agent 生成可组合的排序策略程序
│   ├── dataset.py            # 数据划分、HR/NDCG/ARP/APLT/PopREO
│   ├── candidates.py         # LLMRank 官方 Random/SASRec 候选
│   ├── retrieval.py          # Amazon 自建 Title-BM25 hard negatives
│   ├── amazon.py             # Beauty/Sports/CDs 数据预处理
│   ├── tools.py              # 只读诊断与配对指标证据
│   └── report.py             # 输出实验对照
├── research/
│   ├── scientific_agents.py  # 原 RSI 的研究 Agent / 证据权威
│   ├── scientific_skills.py  # 原 RSI 的技能证据生命周期
│   └── scientific_tree.py    # 原科学树入口（兼容旧任务）
└── runtime/                  # vLLM、JSON Schema、硬预算与日志
configs/llmrank_qwen.json     # 本地 Qwen3-8B 固定低温测量
configs/llmrank_open_policy_qwen.json # Designer 探索采样与排序测量隔离
datasets/llmrank/             # 处理好的 ML-1M/Games
datasets/amazon/raw/          # Amazon 2014 原始压缩文件及续传缓存
datasets/amazon/processed/    # 已校验/预处理的 Amazon 5-core 数据
runs/                         # 真实模型实验和冻结科学证据
tests/                        # 新旧科研协议的回归检查
```


## 数据与实验实际进度

完整、可核验的状态见 [docs/RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md)。

- **ML-1M、Games：** 官方处理数据及 Random/SASRec 候选已接入；Qwen 实验已执行。
- **Amazon Beauty：** 2014 5-core 的评论/元数据均已校验，得到 22,334 用户、12,094 商品、198,255 次交互；已完成一组 24 用户、两轮 RSI、最终留出评价。
- **Amazon Sports：** 原始两份 gzip 均已校验，得到 35,359 用户、18,267 商品、294,151 次交互；已完成24用户、两轮真实 Qwen 实验。
- **Amazon CDs：** 已完成源文件完整性校验、52,808 用户/49,866 商品/814,068 交互的转换，并有两轮真实 Qwen 实验。

Amazon 的 `--retrieval random` 是冻结随机负例，`--retrieval bm25` 是新增的冻结标题检索 hard-negative 协议；二者都不是 LLMRank 官方提供的 Amazon 召回候选。**不能将不同候选协议的数据直接作为同一实验条件比较。**

## 如何运行

先确保一个**自己有权使用的** GPU 上运行 Qwen vLLM 并检查 `configs/llmrank_qwen.json` 中的端口。本命令不会自行启动模型，也不会停止其他人的计算任务。

```bash
cd /data/cjh/rsi/project
PY=/data/cjh/rsi/envs/research/bin/python

# 跨轮次自主研究：同一开发人群递进搜索，测试集最后仅打开一次
$PY -B -m rsi_researcher llmrank-cycle \
    --config configs/llmrank_qwen.json \
    --data datasets/llmrank --dataset ml-1m \
    --users 12 --seeds 20261025 \
    --max-rounds 2 --max-nodes 1 --patience 2 \
    --requests-per-round 130 --max-total-requests 350 \
    --output runs/UNIQUE_CYCLE_NAME

# 单批次的开发 + 留出集审计
$PY -B -m rsi_researcher science-tree \
    --task llmrank_bias --config configs/llmrank_qwen.json \
    --data datasets/llmrank --dataset Games \
    --cohort-size 30 --max-nodes 2 --max-requests 350 \
    --output runs/UNIQUE_PILOT_NAME


# Beauty 的 Title-BM25 检索协议（更困难的候选池）
$PY -B -m rsi_researcher llmrank-cycle \
    --config configs/llmrank_qwen.json \
    --data datasets/amazon/processed/Beauty --dataset Beauty \
    --retrieval bm25 --users 24 --seeds 20261030 \
    --max-rounds 2 --max-nodes 2 --patience 2 \
    --requests-per-round 320 --max-total-requests 850 \
    --output runs/UNIQUE_BEAUTY_BM25

# 对 Sports/CDs 改成 --data datasets/amazon/processed/Sports 或 CDs，
# 但先验证数据来源已经下载及预处理完成。

# 测试
$PY -B -m unittest tests.test_llmrank_mainline tests.test_llmrank_cycle -v
```

**这些运行命令是可运行协议，不表示示例的 100 用户 × 3 种子已经完成。** 目前仅有较小样本的真实 Qwen 验证；不能因为开发集 ARP 下降而宣称找到成功的去偏算法。

## 开发规范

- 只有 `dataset.py` 的独立评价器决定指标，Agent 不能读取隐藏标签或者修改科学阈值。
- Critic 只审查准入合法性；实际质量、偏差、成本分别记录。
- 保留每次失败的反证、对应源码/策略、原始模型输出和预算。
- **内层树搜索可以多节点，外层 Cycle 可以多轮；都必须有总预算、Patience 及独立确认。**
- 更大策略空间不意味着放任 Agent 任意执行代码。新增机制请实现为可复现实验算子再开放给 Designer。
- 历史主线由版本记录恢复。目录整理只移除过时的开发工作区，不删除原始科学记录或其他会话进程。
