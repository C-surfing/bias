# Open-Ended Policy Programs for RSI Debias

研究主线仍是推荐系统去偏，不把 Harness 自我改写变成独立研究目标。
正式实现：`rsi_researcher/llmrank/{program.py,formula.py,strategy.py,agents.py,science.py,cycle.py}`。
直接运行入口：`python -m rsi_researcher llmrank-cycle`。

## 为什么不继续枚举更多动作

早期动作只有 history_n / history_order / candidate_order / votes /
popularity_lambda / rerank_policy / score_expression，公式只使用
`base,pop,match`。这是参数组合，不等价于自主算法发现。

- 多数 Amazon 5-core 用户训练历史过短，history_n=10/20/50 在
  同一批用户上等价；扩大枚举动作产生大量重复无效实验。
- 对所有人统一减少热门商品得分，可能减少真实相关商品；
  没有近期兴趣与用户流行度画像，无法进行有条件的去偏。
- 仅靠候选重排不能修复召回集里缺少相关长尾的情况。
- 14 名开发用户 + 10 名留出用户，一个正例名次变化就可能改变
  NDCG/HR 结论；需要保留可执行但未通过的机制证据，不能立即判定
  整个方法族失败。

## 已实现（阶段 A：受控开放式程序）

Agent 可以输出任意合法 JSON `policy_program`，含 1..5 个类型化
阶段，而非从几十个预制策略名称中挑选。可自由推导连续系数
`popularity_lambda∈[0,2]`，并组合非线性条件表达式。
具体示例（仅演示语法，不声称有效）：

```json
[
  {"op":"score",
   "expr":"score - (0.22*pop_excess if recent_match < 0.2 else 0.03*pop)"},
  {"op":"mmr","relevance_weight":0.87},
  {"op":"protect","top":2}
]
```

支持的数值特征由固定训练记录和真实 Qwen 排名计算，
不依赖目标正例：`base`, `score`, `pop`, `match`,
`recent_match`, `user_pop`, `pop_excess`, `novelty`,
`head`, `rank_norm`, `history_len`, `pop_percentile`。

编译器允许 `+ - * /`、`min/max/abs/clamp/sqrt/log1p`、
`if ... else ...` 单条件、有限常数，执行时只有完整候选排序，
不得访问 target、文件系统、网络或执行 Python import。
多阶段可以依次更新分数、MMR 去同质化、保护部分原排名、
限制候选内的热门项。后续每一种新机制可以只增加一处语义明确的
编译器算子，不增设重复 Agent 或评价器。

`history_n=20` 和 `history_n=50` 在最多只有 19 条历史的
开发集上会合并为同一**功能指纹**，不是重复花费搜索预算。
程序字符串经规范化 JSON 后纳入策略指纹，实验文件保存完整代码、
Critic 准入及独立数值评价。
`ranking_parser_contract=rank_alias_v2_open_policy_program_v1` 用于隔离
旧实验，不能将上一协议失败结果当成同一实验的直接比较。

### 当前边界：依然不是任意 Python 政策编程

Agent 现在可以设计 **程序结构 + 连续数值 + 分段条件**，但底层
特征和四种阶段语义仍由可信执行器定义。算法族显著拓宽，但尚未
达到「写任何推荐算法」。这是第一步，不应夸称开放世界算法发现。

下一阶段建议支持 Agent 在独立隔离的实验工作目录里编写
`policy.py`，它只需实现：

```python
def rerank(user_history, candidate_features, llm_ranking, train_stats) -> list[str]:
    # Agent may implement arbitrary candidate-preserving reranking.
    ...
```

但要有真正 OS/容器进程边界：
- 只读训练历史/商品标题/Qwen 冻结预测；无标签、无 holdout。
- 不允许修改数据加载、Qwen Gateway、评估器、科学门槛与策略选择报告。
- 在无网络、禁写项目、独立 uid、限时限 CPU/内存的隔离容器运行。
- 输出恰好为候选集合的一次全排列，并记录源代码 SHA256 与资源成本。
- 先在同一冻结开发集上比固定策略/随机搜索，唯一一次独立确认后
  才采纳发现；失败也作为负面证据写入跨轮记忆。

**不能直接把任意生成的 Python `exec()` 到主 RSI 进程里。**
那样 Agent 连保护边界和科学评估都能绕开，无法形成可信研究。

## 科研创造与推荐测量分离

`configs/llmrank_qwen.json` 使用温度 0/固定种子，适合推荐结果配对，
但 Designer 同时温度 0，导致两轮树搜索重复相同代码。
现在支持 `configs/llmrank_open_policy_qwen.json`：
- Ranking / Critic 等科学测量继续固定 `temperature=0`；
- 只有 `scientific_role=design` 可以 `temperature=0.75`、
  `top_p=0.95`，且种子由实验 episode 标识 SHA256 确定；
- 新采样元数据随实验保存，排名缓存不能跨不一致配置混用。
这不是为了产生更多随机文本，而是防止策略机制探索收缩成同一模板。

启动示例：
```bash
cd /data/cjh/rsi/project
/data/cjh/rsi/envs/research/bin/python -m rsi_researcher llmrank-cycle \
  --config configs/llmrank_open_policy_qwen.json \
  --data datasets/amazon/processed/Beauty --dataset Beauty --retrieval bm25 \
  --users 24 --seeds 20261104 --max-rounds 2 --max-nodes 2 \
  --requests-per-round 320 --max-total-requests 850 \
  --output runs/UNIQUE_OPEN_PROGRAM
```

**已完成的第一批 open-program 实验证据：**
`runs/beauty-open-policy-v1-normalized-20261103/` 的 Agent 实际提出
分阶段 `score+mmr+protect+head_budget`，开发集 ARP 24.27→14.84，
NDCG 0.0727→0.0615，因此不通过 5% 效用容限；并且温度 0 的
Designer 多次重复同一策略。这是程序执行成功、科学 Debias 仍失败，
不能借“空间变大”直接宣布找到算法。

## 需要什么才算策略空间真正帮助了 Debias

先用不同的冻结用户组，在相同 Qwen 调用/Token 预算下做三组消融：

1. 旧离散策略枚举 + 相同 Tree 深度。
2. 当前自主程序表达式 + 有限多阶段。
3. 稳定实现后，隔离代码生成式策略。

固定：数据划分、候选检索来源（Amazon Random 或 BM25，各自一组）、
模型温度/版本、指标定义、开发/留出集合。报告新增机制数量、
独立通过质量门槛的 Debias 次数、NDCG/HR/ARP/APLT/位置偏差的
配对 CI、有效排序与修复率、总模型调用/Token 成本。
多轮研究只有在开发阶段改进候选，不得反复使用测试标签挑胜者。

**已经确认的经验事实（不是本程序新实验结果）**：
Beauty 原 BM25 + parser v2 的开发集曾出现通过联合门槛的 RSI
候选，但留出集 ARP 下降而 NDCG 退化；Sports、CDs 旧空间也均
反复出现这种质量/去偏权衡。新程序的有效性必须实测，不能
因代码可运行就判定解决了推荐偏差。

设计参考：
- https://github.com/karpathy/autoresearch
- https://lilianweng.github.io/posts/2026-07-04-harness/
