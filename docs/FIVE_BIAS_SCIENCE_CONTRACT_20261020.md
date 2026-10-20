# 五类 Debias：POPPER 反证契约与 AutoDebias 式策略优化实测（2026-10-20）

研究工作树：/data/cjh/rsi/experiments/rsi-multiagent-20261017-chatgpt。只改独立工作树，不改 main、别人的进程或历史结果。

## 官方实现借鉴

AutoDebias: https://github.com/DongHande/AutoDebias/blob/main/train_explicit.py
官方包含 MetaMF、weight1_model、weight2_model、imputation_model、模拟单步更新与随机参考损失的元目标。当前不引入元梯度：借鉴的是随机评分参考作为策略参数 alpha、beta 搜索的外层损失。

POPPER: https://github.com/snap-stanford/POPPER/blob/main/popper/agent.py
以及 https://github.com/snap-stanford/POPPER/blob/main/popper/popper.py
借鉴主零假设、可观察反证与逻辑反例的明确映射；不复制动态生成和 exec 的代码，也不将重复读取 Coat 开发集的 p 值进行无效累乘。

## P0：科学结论的唯一权威来源

scientific_agents.authoritative_evidence() 是原科研循环与五类 Debias Director 共用的数值证据语义。程序标记 statistical_significance=NOT_ESTABLISHED_BY_THIS_WORKFLOW、sequential_error_control=NOT_AVAILABLE、确认发现数=0。Reflection 的自然语言只保留为未经裁决的后续研究假设，不得作为显著性/因果/独立确认的证明。测试故意使用 FakeModel 输出 statistically significant 和 confirmation not required，也无法改变数值裁决。

历史事实：Selection cap=100 通过开发 Brier，但事前冻结的 2209 条保留集联合质量验收失败（NDCG 相对基线 98.07%，低于冻结的 99%）；保留集只使用了一次并且不能再用来调参。本轮没有打开它。

## P1：零假设、可证伪性与 Critic

five_bias_development.falsification_protocol() 对五个经典 Bias 统一输出:
primary_null_hypothesis / registered_falsifier / required_observations /
missing_or_nonidentified_if / admissible_conclusion / counterexample_challenge。

原 Critic Agent 增加五类研究专用的结构化 review_debias：必须说明主零假设成立却看到指标改善的具体反例、尚未排除的竞争机制、已注册反证是否可观测以及与零假设之间的逻辑关联。如果不相关或不可观测，Python 在执行前拒绝。通过不是因果鉴定；Exposure 的预测代理不冒充 OPE，Popularity 的头部份额下降不冒充不合理热门偏差，Filter Bubble 的模拟因果不冒充真实用户实验。

## P2：受约束连续组合策略研究

继续复用原 Director 与现有四个 Agent。通过 science-debias 的 --joint-design-only --max-joint-proposals 3，在同一冻结随机评分开发数据上允许最多 3 个不同连续提案，参数范围 alpha∈[0,1]、beta∈[0,.15]。

组合公式：
- score=(1-alpha)*unweighted_item_score+alpha*estimated_propensity_item_score
- adjusted_rank=score-beta*log1p(train_item_count)/log1p(max_count)

冻结验收：Brier gain≥0.002；NDCG≥未加权基线99%；Hit下降≤0.02。每轮被拒绝、重复参数或质量失败都有状态；合法方法通过则结束开发，模型调用不足或尝试次数用尽也会停止。不会无限重试到偶然显著。

运行记录附带真实 CPU 数值计算时间、墙钟时间、模型调用次数、Critic 拒绝和质量失败。代码另把重复设计拒绝提前到 Critic 调用前；旧 Qwen 记录保持原样。

## 本次真实 Coat 开发数据对照（有意保留负结果）

固定规则连续搜索：2 轮 / 0 次 LLM / 2 次实际数值评价 / 0.335 秒：
- alpha=.75、beta=.05：Brier gain +0.00429754，但 Hit 0.93143，质量保护失败。
- alpha=.90、beta=0：Brier gain +0.00407965；NDCG 0.711590；Hit 0.942857，开发目标通过。

本地 Qwen3-8B 真实运行：3 轮 / 15 次 LLM / 1 次实际数值评价 / 58.2 秒：
- alpha=.70、beta=.10：Brier gain +0.00429036；NDCG 0.707338；Hit 0.931429，质量保护失败。
- alpha=.60、beta=.12：Critic 判定实验不相关，拒绝，未进行数值评价。
- alpha=.60、beta=.12：重复参数，被数值工作流拒绝；此轮运行后已修复无意义的重复 Critic 成本。

原始文件：
runs/rsi-joint-rule-sequential-contract-20261020-01/director_state.json
runs/rsi-joint-qwen-sequential-contract-20261020-01/director_state.json

结论：在这个开发问题上，强固定规则更快且达到开发目标；真实 Qwen 没有证明科研效率或去偏效果优势。上述数值不是独立确认，没有所谓统计显著；没有发明任意新代码，只是在安全的策略组合语法中提出和检验新候选。

## P3 与剩余真正科学工作

旧 Coat 保留集已消耗，不能再用；新候选必须先冻结再用完全未见的用户/时间/数据源独立确认。其余四类仍有不同可识别性边界：真实 KuaiRand 部署倾向/支持集不全；Popularity 机会基准只来自有限随机候选；Filter Bubble 的多轮反事实现阶段是有已知潜在结果的模拟环境；Fairness 的当前目标是 Coat 二元 gender 的预测误差差距，不代表 Equal Opportunity。

Tree/Linear/Rule/no-history/audited-history/shuffled-history 在同预算、未见任务上的跨问题比较尚未完成。本轮没有声称解决 P3，也不会为追求正结果继续对已看过的随机评分保留集进行测试。独立确认 Debias 发现总数仍为 0。
