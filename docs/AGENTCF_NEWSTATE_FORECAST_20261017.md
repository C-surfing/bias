# 新用户 AgentCF：结构化预测与反证（2026-10-17）

## 已执行而非计划中的事项

- 从旧 20 用户 AgentCF 实验的原始输入提取实际 user ID，训练数据剔除他们，抽取六个其他用户、每人两条反馈。重新调用 Qwen 的 AgentCF.event 完成十二次训练事件，创建新 User/Item Memory 状态。
- 将六个用户的 validation 查询均分为 A/B 各三人；固定候选和用户状态，实际分别运行原始排序、重复排序、候选 Item Memory 复原和只修改流行度指令的排序。
- Qwen 两组（无历史、仅经来源验证的历史）分别针对 Item Memory 与 Popularity 提交两项机器可评分的指标方向、最小效应、竞争解释、反证和质量边界。所有预测在状态训练之前冻结并保留 SHA-256。
- Python 独立计数预测是否在两个新面板都吻合且 NDCG@5 不低于基线 99%；绝不由 Qwen 自评代替数值审查。

## 已测结果

| 策略 | 正式预测 | 同时通过两组的预测 | 独立确认科学发现 | 规划+复盘请求 |
|---|---:|---:|---:|---:|
| 强规则 | 2 | 0 | 0 | 0 |
| Qwen 无历史 | 2 | 0 | 0 | 4 |
| Qwen 来源验证历史 | 2 | 0 | 0 | 4 |

Item Memory：A/B 的去噪排名变化率分别 0.000 / +0.667；
NDCG 差值 -0.129 / +0.210，面板 A 质量门槛失败。
Popularity：两组的热门超额曝光变化都约为 0；
带记忆和无记忆的 Qwen 对指标方向的预测相反，但均没有通过。

## 真实计算与科学边界

- 全部 92 次 Qwen3-8B 调用，包含规划、十二个新反馈训练事件、24 个重新排序与研究复盘。
- 78,030 prompt tokens + 11,607 completion tokens；约 191 秒。
- 单独 GPU5 vLLM 实例正常退出，最后 GPU 显存约 14 MiB；未访问或修改独立确认数据。
- 此实验与旧 cohort 不同用户，但仍是同一作者 CDs 数据来源、相同机制类别，
  同一新训练状态、每面板仅三人，不能证明不存在机制，更不能宣称 RSI 提升。
- 已知目标只有一个验证 positive item，其他候选的真值未被观察。
- 结论是所有预测在严格跨面板数值门槛下失败，独立科学有效发现=0；绝不算入论文显著改善。

## 原始证据位置

唯一主工程：/data/cjh/rsi/project/
源代码：rsi_researcher/research/fresh_predictions.py、fresh_runner.py、fresh_audit.py、fresh_memory.py
结果：runs/agentcf-newstate-forecast-20261017/
包括 protocol.json、protocol.sha256、predictions.json、forecasts_frozen_at.json、
frozen_state.json、ranking_panels.json、measured_numeric_evidence.json、
machine_scored_forecasts.json、INDEPENDENT_AUDIT.md、calls.jsonl、
SOURCE_CHECKED_RESEARCH_MEMORY.json。

## 下一次最有价值的研究实验

扩大到多个全新用户 cohort 与不同 AgentCF 训练种子，增加每组查询数；
要求 Qwen 在每次研究前提交正式效应方向和阈值，与强规则及空历史对照
在同等计算预算下比较**独立验证的有效发现数/模型请求与 GPU 成本**。
增加共享 Item Memory 写入路径与下游读出顺序的随机化或因果对照，
并独立校验误记忆文本，避免仅靠热门占比和已知目标 NDCG 归因。
