# RSI 科研完成度与实验证据（2026-10-21）

> 只认可实际文件、真实 Qwen 调用、独立评价器输出。这里不将功能实现、数据准备、正式科研成功混为一谈。
> 研发唯一目录为 `/data/cjh/rsi/project`；不是原型 worktree 或 `experiments/`。

## 数据集核验

| 数据集 | 输入来源 | 数据准备 | 真实推荐/RSI | 可用的科学结论 |
| --- | --- | --- | --- | --- |
| ML-1M | LLMRank 官方已处理输入及 Random/SASRec 候选 | ✓ | ✓，数次小样本实验、2轮循环 | 无独立确认的有效 Debias |
| Games | LLMRank 官方预处理及 Random/SASRec 候选 | ✓ | ✓，小样本实验 | 无独立确认的有效 Debias |
| Amazon Beauty | Amazon 2014 5-core review + metadata | ✓，源文件验证与 provenance | ✓，24用户、2轮真实 Qwen 实验 | 新策略未通过联合门槛 |
| Amazon Sports | Amazon 2014 5-core review + metadata | ✓，源文件验证与 provenance | ✓，24用户、2轮真实 Qwen | 新策略未通过联合门槛 |
| Amazon CDs | Amazon 2014 5-core review + metadata | **未完成**：评论源文件仍在续传 | **未运行** | 不可声明 |

### Amazon 准确规模

| 类别 | 原始 5-core 评论 | 保留用户 | 保留商品 | 有效交互 |
| --- | ---: | ---: | ---: | ---: |
| Beauty | 198,502 | 22,334 | 12,094 | 198,255 |
| Sports | 296,337 | 35,359 | 18,267 | 294,151 |
| CDs | 1,097,592（官方原始规模） | 待处理 | 待处理 | 待处理 |

Amazon Beauty / Sports 的数据以 `datasets/amazon/processed/{Beauty,Sports}/` 保存
`*.inter.zip`、`*.item` 及 `provenance.json`（含 SHA256、标题缺失统计、筛选规则）。

**数据混淆禁令：** 仅在 ML-1M/Games 使用 LLMRank 作者预计算召回候选；
Amazon 的 `random` 是研究者冻结随机候选，`bm25` 是基于训练历史的标题
检索 hard negatives，不是作者官方 Amazon 结果。不同召回来源的比较必须分协议报告。

## 已实装的科研流程

- [x] 统一 `python -m rsi_researcher`，源码直接在正式项目目录
- [x] 真实 Qwen LLM 推理和完整排名校验
- [x] Diagnosis/Designer/Critic/Reflection 复用原科研 Agent 体系
- [x] 可组合历史/候选位置/Prompt/投票/安全公式/全局与条件性去偏及 top-3 保护
- [x] 内层树节点的父子关系、量化数值审计、实验成本和失败记忆
- [x] 外层多轮循环的逐轮记忆继承、策略精确去重、预算、patience
- [x] 相同 system/prompt/schema/模型元信息跨轮 Qwen 响应冻结复用，避免无效波动
- [x] 中间轮次 development-only，最终选择冻结后留出集只评估一次
- [x] HR/NDCG/ARP/APLT/PopREO/Candidate Position Bias 与配对 bootstrap 区间
- [x] Amazon Beauty/Sports 数据格式统一与源文件溯源
- [x] Amazon 冻结 Title-BM25 候选检索和随机候选严格区分
- [ ] 三个 Amazon 数据集都完成真正的 Qwen 端到端实验
- [ ] >=200 用户、>=3 种子、多候选检索协议、独立新用户统计确认
- [ ] 与固定策略、随机搜索、Agent 搜索按真实调用/Token **严格同预算**比较
- [ ] 完整的原论文 RecBole 用户划分、原始生成提示及原模型逐项复现
- [ ] 新算法级干预算子（如质量约束交换/个性化校准/曝光反事实）科学验证
- [ ] 在独立数据集复现偏差改善且效用不退化

## 已完成真实 Amazon Beauty 研究

`runs/beauty-qwen-real-multiround-20261028/`：
24 位用户，14 development、10 holdout，2轮研究，共210次真实 Qwen 请求，
其中最后20次为独立确认阶段。第一轮开发集：

| 策略 | HR@5 | NDCG@5 | ARP@5 | APLT@5 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.3571 | 0.2168 | 14.2545 | 0.8000 |
| Agent N001 | 0.4286 | 0.2507 | 19.6727 | 0.7818 |
| Agent N002 | 0.2143 | 0.1736 | 13.1800 | 0.8400 |

N001 质量改善但热门程度更高；N002 热门程度下降但质量受损且排序有效性退化。
两轮后 RSI 均未通过联合验收，留出保持原始策略：
HR@5=0.2000，NDCG@5=0.1631，ARP@5=24.625，APLT@5=0.7000。
PopREO 留出指标 N/A（缺少正例组支持），不能当作 0。



## 已完成真实 Amazon Sports 研究

`runs/sports-qwen-real-multiround-20261029/`：
24 用户，14 development / 10 holdout，2 轮研究，共 206 次 Qwen 请求。
开发集 Baseline NDCG@5=0.2236、ARP@5=8.875、APLT@5=0.8500。
第一轮 Agent 将 NDCG 提升至 0.3780，但 ARP 增至 17.40；第二轮
将 ARP 降至 6.175、APLT 提升到 0.975，却把 NDCG 降至 0.1429。
没有候选同时通过效用和偏差的联合验收。最终留出仍选择 Baseline：
HR@5=0.4000，NDCG@5=0.2931，ARP@5=29.8889，APLT@5=0.8444。

**两种 Amazon 随机候选环境中 Baseline APLT 都已很高**，
随机负例本身可能造成容易命中的长尾候选池，导致流行度代理改善的
上限很小。新增 Title-BM25 检索属于更难的候选对照环境；不可
与 LLMRank 作者预先计算的 ML-1M/Games 候选协议混为一谈。

## 下一阶段按科学价值排序

1. 等待 CDs 2014 源文件完整、复核 gzip/sha256 和转换；**未完成前禁止报告 CDs 结果**。
2. 对 Beauty / Sports 各自冻结同一召回协议，比较 Random 与 Title-BM25 在推荐
   质量、真实热门程度与去偏 Pareto 上的不同。此处比较的是环境变化，不要
   将 BM25 检索变好直接当作 Agent Debias 效果。
3. 增大开发与留出用户数；分别报告跨用户收益、方差、长尾/热门兴趣分群与
   Pareto 前沿，不再用小样本单轮数字声称跨用户稳定。
4. 收敛策略语言到真正会改变推荐机制的条件性算子，不把自由 Prompt 的细微
   改写当作全新研究节点；重复失败族采用**软剪枝**，有强重复证据后才硬剪枝。
5. 冻结批次、多种子、严格同调用预算的 Random/Fixed/RSI 搜索对照，然后独立
   confirmation；只有这些成立才能宣称自主发现有效去偏策略。
