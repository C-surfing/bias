# 五类 Bias → Debias 的唯一研究主线（2026-10-20）

## 项目目标和目前真正完成的部分

**研究目标不是证明 Multi-Agent 会提出更多记忆假设，而是证明它能自主发现、反证、解决五类推荐偏差，并在未见任务上提高单位研究成本的独立确认去偏结果。**

唯一的 Bias 定义来自 `rsi_researcher/research/bias_registry.py` 及其冻结的 `configs/bias_tasks.json`。原来科学树搜索的四个任务是**实验环境**，不是另一套五类偏差。尤其 `agentcf_memory` 只研究共享 Item Memory 的传播机制，它不是第六种 Bias，也不代替 Filter Bubble 或 Fairness。

| 五类目标 | 对应当前代码 | Debias 方法与执行现状 | 识别/确认缺口 |
|---|---|---|---|
| **Selection** | `coat_selection.py`; `coat_propensity`, `coat_utility` | 在真实 Coat 用相同容量的 item-mean 比较无权重与估计 propensity 的 clip=5/20/100；评估随机评分开发集 | 作者 propensity 是估计值；Hit@5 代价；没有独立确认 |
| **Exposure** | `bias_registry.py` 和 KuaiRand / controlled 环境诊断 | 尚未建立可执行的可信曝光校正与 OPE 完整链 | 需要曝光日志、已知/可识别倾向、positivity 和时间对齐 |
| **Popularity** | `agentcf_popularity`, `scientific_experiments.py` | Prompt 减弱热门描述 + 同状态排名 / 头部曝光代理变化 | 无法从代理的头部曝光变化推出有害的因果流行度偏差被纠正；独立相关性不足 |
| **Filter Bubble** | 冻结任务定义与部分交互环境 | 未完成多轮曝光—反馈—推荐轨迹的受控 Debias | 需要时间顺序、主题标签、相同初态和相关性保护；单次多样性不成立 |
| **Fairness** | 冻结任务定义 | 尚无可信群体约束重排与公平评价闭环 | 必须先决定合法且科学合理的群体、公平准则和机会/效用参照，热门/长尾不自动是社会公平性 |

## 本轮真实 Debias 开发对照：Coat Selection

执行 `rsi_researcher.research.coat_selection`，来源：`runs/coat-fivebias-selection-debias-20261020-01/coat_selection_development.json`。

- 6960 条非随机训练评分；2065 条随机评分开发；2209 条随机评分独立保留集**未用于评分**。
- 模型形式、平滑先验与容量固定；只改变训练中观测评分的估计 propensity 倒数权重及截断上限。
- 未加权：Brier **0.153281**、NDCG@5 **0.709910**、Hit@5 **0.954286**。
- clip=100：Brier **0.149546**、NDCG@5 **0.710947**、Hit@5 **0.942857**。Brier 下降约 0.003735，NDCG 略升、Hit 明显下降。不能称“全部效用都改善”。
- 选择 clip=100 是**开发数据驱动**；不能复用开发数据作独立确认。原始倾向仅为估计值，不是平台真随机曝光概率。
- 所有证据仍为 `DEVELOPMENT_COMPLETE_NO_CONFIRMATION`，独立确认科学发现数为 0。

## 统一、不过度工程化的架构

```text
五类 Bias Registry（问题定义、可识别性、目标指标）
                 ↓
真实数据 + Support Gate（证据是否足够）
                 ↓
Diagnosis → Designer → Critic（提出可区分的实验）
                 ↓
实验注册动作 → 原有推荐执行器（不能任意修改评价器）
                 ↓
independent_verdict（唯一数值审计）
                 ↓
classify_debias_evidence（同一审计模块里的科学资格语义）
         ↙                ↓                 ↘
机制诊断          代理指标变化         Debias 开发候选
         ↘                ↓                 ↙
Research Memory（只保存可追溯数值证据）
                 ↓
Tree（单任务分支）→ Director（跨任务选择、预算与验收）
                 ↓
独立确认（尚未实现完整统计/强基线；不可自行宣称）
```

没有新 Agent、另一套 Orchestrator、第二个数值评价器或重复的 Bias Registry。Director 不再根据自身临时拼接的 `forecast=True + quality=True` 公式授予 Debias 候选资格；**唯一可信口径属于原有 scientific_experiments 独立审计**。

### 旧版真正的验收接口 Bug

Coat 动作级反证来自冻结 `reference.assess()`，但 AgentCF 的 `registered_action_falsifier_passes=None`。旧 Director 严格要求该字段是 `True`，同时要求 Coat 与 AgentCF 两个家族：默认开发期资格在真实路径上不可达。

**修复原则**：不把 AgentCF 的 `None` 假装改成 `True`，也不把 Item Memory 或热门代理计为 Debias。现在：
- `selection` 的真实 `split_utility` 动作，必须效应/注册反证/NDCG 保护符合冻结条件才有开发候选资格；`gamma_stress`、`propensity_shuffle` 是**反证控制**，不是 Debias 方法；
- `popularity` 的 Prompt/头部比例，只能记代理变化；
- `agentcf_memory` 一律记辅助机制；
- `exposure`、`filter_bubble`、`fairness` 未具备实验闭环就显式标记缺口；
- `decide_acceptance()` 使用五类目标计数，不会把两个 Coat 子任务误计为两个 Bias 家族；跨两种 Bias 默认验收现在会明确标记“缺少已识别 Debias 方法”，不能成为一个隐式永远不满足的门槛。

## 下一轮只应该做的工作（不是新架构）

1. 在 Selection 上使用未见数据与固定策略进行真正的**独立确认**，包括 Brier、NDCG、Hit 及成本；补 IPS/DR/AutoDebias 强基线并清楚注明估计倾向假设。
2. 为 Exposure 完成日志支持和可信倾向审查；若 KuaiRand 不能支撑用户期望的目标策略，选择合法受控环境验证，**不伪造因果证据**。
3. 为 Popularity 增加独立相关性/机会参照的目标方法（如支持保持的重排），比较头部曝光变化与质量损失；单靠 Prompt 去除热门描述不合格。
4. Filter Bubble 必须做多轮同初态轨迹对照；Fairness 必须先注册群体及规范性公平定义，不能拿热门/长尾比例冒充公平结果。
5. 待两类以上可信 Debias 方法有开发资格、并建立独立未见任务后，再严格比较 Tree/Linear、Audited/Shuffled Memory 和 Research Skill 的额外科学贡献与 Token/GPU 成本。

**原则**：Debias 结果优先于 RSI 搜索强度；可识别性优先于参数优化；每个结果有可证伪的指标和失效边界；减少无价值模块，而非增加漂亮的架构图。
