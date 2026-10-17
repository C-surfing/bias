# 合成 Exposure RSI 科研决策对照：2026-10-17

## 科学问题与限制

Qwen3-8B 的历史经验检索及研究策略修订，能否超越强固定规则？
这次只检验合成 Exposure OPE 方法选择；未检验真实推荐去偏、真实科研发现或完整 RSI。

数据：7 种开发机制，每类一个开发配置；同机制评价每类两个未见配置，共 14 题。
模型看到日志诊断，真值与评价重复随机种子仅供独立 Python 评分器使用。
动作空间：naive、IPS、SNIPS、abstain。错误因果行为由程序化约束拒绝。
每模型组 15 次 Qwen 请求（1 校准 + 14 决策），合计 60 次；规则模型 0 次。
模型组请求数相同，但 prompt tokens **不同**。

## 已保存结果（mean regret 越低越好）

| 策略 | Qwen 请求 | Mean regret | 不必要退出 |
|---|---:|---:|---:|
| 强规则 | 0 | 0.001141 | 0 |
| Qwen 无历史 | 15 | 0.001263 | 0 |
| Qwen 相关历史 | 15 | 0.001917 | 1 |
| Qwen 策略修订 | 15 | 0.002474 | 1 |
| Qwen 打乱历史 | 15 | 0.005275 | 4 |

当前是负结果：没有发现相关记忆/策略修订优于强规则的证据。
打乱历史增加无必要弃权，但这一观察还不是检索污染的因果效应证明。
所有组的无效因果执行为零，但安全闸门贡献不能算作模型理解能力。

## 复现与冻结

- Frozen protocol: runs/decision-benchmark-v1-20261017/frozen_protocol.json
- SHA-256: 4baaf91299e8b98b75a91644e6706c7b78da3cecfd49a09a70bbb2925978317b
- 原始决定、校准反馈、成本：runs/decision-benchmark-v1-20261017/qwen-pilot-01/
- 独立研究复核：上面子目录的 ANALYSIS.md / ANALYSIS.json
- 实际实现：rsi_researcher/research/decision_benchmark.py、decision_qwen_pilot.py、
  decision_analysis.py。
- 模型已退出，GPU5 最后 14 MiB。
- 最优动作由保留种子的 24 次 Monte Carlo 风险最小值近似产生；
  不是理论 Bayes 最优动作。
- 7 类机制的聚类 bootstrap 统计力很低，不能断言显著性。
- 这 14 个已评价配置不得再当作未见测试数据。

## 下一步

1. 以跨机制的新任务（Selection/Popularity/AgentCF Memory）测试真实研究发现能力，
   而非仅要求模型选择 OPE estimator。
2. 预注册、随机化 Arm 和任务顺序，冻结调用次数及 token 上限。
3. 比较不使用历史、来源可靠历史、错误历史及策略修订；加入
   错误记忆的矛盾检测，保留完整失败记录。
4. 在已验证的机制真值基础上评价发现/成本/推荐质量，而不使用这一轮调整成功指标。
