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
| Amazon CDs | Amazon 2014 5-core review + metadata | ✓，源文件校验与溯源 | ✓，24用户、两轮真实 Qwen | 没有独立确认的有效 Agent 去偏 |

### Amazon 准确规模

| 类别 | 原始 5-core 评论 | 保留用户 | 保留商品 | 有效交互 |
| --- | ---: | ---: | ---: | ---: |
| Beauty | 198,502 | 22,334 | 12,094 | 198,255 |
| Sports | 296,337 | 35,359 | 18,267 | 294,151 |
| CDs | 1,097,592 | 52,808 | 49,866 | 814,068 |

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

## 已完成 Amazon CDs 真实 Qwen 小样本研究

`runs/cds-qwen-real-two-round-20261101/` 记录 24 用户、
两轮开发搜索与唯一一次留出确认。开发集 Baseline
NDCG@5=0.28698、ARP@5=20.94；固定 Popularity 策略
ARP@5=15.64 且 NDCG 不变，但留出没有延续。
最终 RSI 选择原始策略，留出 HR@5=0.20、NDCG@5=0.13869、
ARP@5=17.74、APLT@5=0.66。无可确认的自主去偏发现。

## 开放策略程序（新增）

开发集研究可以由 Qwen 自主编写 1..5 阶段的 `policy_program`，
组合用户条件公式、MMR 多样性、rank 前缀保护、head 配额；
包含连续参数、近期历史匹配、用户流行度画像和非线性表达式。
代码在 `program.py`、`formula.py`、`strategy.py`。
此方式尚不支持任意 Python 算法，详细边界见
`docs/OPEN_POLICY_RESEARCH.md`。
所有程序都由可信解释器运行，继续冻结召回、标签和指标。
与旧 parser-v2 不混用实验结果，验证仍需多种子同成本对照。

## Open-program v1 探索性实验

`runs/beauty-open-policy-v1-normalized-20261103/`：Qwen 提出了由
score、MMR、protect 和 head_budget 四阶段构成的程序。
开发集 Baseline (NDCG@5,ARP@5)=(0.07270,24.27)，
程序 (0.06153,14.84)。程序能真实改变排名、降低 ARP，但
NDCG 相对损失超过 5%；被实验审计拒绝。
模型配置温度 0 造成跨节点程序大量重复，下一批单独提高
Designer 采样温度、固定 Ranker，结果待确认。

## Harness 格式鲁棒性审计（不改变 Debias 科研主线）

本次只修正科学判断层，**不改变 Agent、树搜索结构、Qwen 参数、推荐策略空间或原始指标定义**。保留包含自动修复结果的完整 NDCG/HR/ARP/APLT，独立记录
`repaired_votes`、`repaired_cases`、`unrepaired_complete_cases`。

新开发集准入还需要一项格式敏感性复核：
- 基于相同用户、相同候选，Baseline/Agent 两种候选位置呈现均未修复的输出形成**配对子集**；不根据真实目标的命中与否筛人。
- 全样本与配对子集均须满足原效用保护及 Bias 改善门槛；干净配对不足 60% 或少于 5 人的候选标记为 **inconclusive_due_to_ranking_format**，不视为成功或确定失败。
- 配对结论相反的候选为 **format_sensitive_candidate**，保留方案、数字和失败机制进入 Reflection 与跨轮记忆，但不作为正式保留策略。
- 树搜索优先级对有效配对使用同一子集评价，对格式支持不足的节点不给虚高的拓展分数。
- 最终独立确认同时保存完整样本、格式质量和配对敏感性结果；**不根据留出结果重新选策略**。
- 评估协议有独立版本 `paired_unrepaired_dev_sensitivity_v1`，不可与旧准入协议混合继承科研结论。

**对 2026-10-31 Beauty BM25 N002 的离线重放：**
完整 14 用户开发集 NDCG@5 0.0451→0.0714，ARP@5 19.03→15.33；
两种位置呈现均未修复且双方一致的用户仅有 8/14，
其 NDCG@5 0.0789→0.0000，ARP@5 18.33→9.43。
这个子集过小，不能断言因果，但足以表明旧的开发集准入
**没有格式鲁棒性证据**；应标记为未确认，而不是有效 Debias 发现。

历史 `runs/` 原始记录未更改。新的格式审查仅适用于新实验，
不把过去的研究结果悄悄按新协议重写。

## 下一阶段按科学价值排序

1. 等待 CDs 2014 源文件完整、复核 gzip/sha256 和转换；已完成源文件与转换核验，但不得将单种子试验作为跨数据集确认。
2. 对 Beauty / Sports 各自冻结同一召回协议，比较 Random 与 Title-BM25 在推荐
   质量、真实热门程度与去偏 Pareto 上的不同。此处比较的是环境变化，不要
   将 BM25 检索变好直接当作 Agent Debias 效果。
3. 增大开发与留出用户数；分别报告跨用户收益、方差、长尾/热门兴趣分群与
   Pareto 前沿，不再用小样本单轮数字声称跨用户稳定。
4. 收敛策略语言到真正会改变推荐机制的条件性算子，不把自由 Prompt 的细微
   改写当作全新研究节点；重复失败族采用**软剪枝**，有强重复证据后才硬剪枝。
5. 冻结批次、多种子、严格同调用预算的 Random/Fixed/RSI 搜索对照，然后独立
   confirmation；只有这些成立才能宣称自主发现有效去偏策略。
