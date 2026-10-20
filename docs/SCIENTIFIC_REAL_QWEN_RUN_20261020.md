# 2026-10-20 独立 Qwen3-8B 科研实验：执行与证据审计

**性质：真实模型驱动的开发期研究测试。没有独立确认，也没有建立 RSI 优越性。**

## 执行范围与数据

- 服务器 worktree：/data/cjh/rsi/experiments/rsi-multiagent-20261017-chatgpt
- 本会话独立 GPU 5、端口 18419、vLLM eager 执行。没有改动/停止他人的 GPU 1–4 任务。
- 实际模型：本地 Qwen3-8B。只有 research 环境的调用日志计入真实模型请求；Python 模型替身的回归测试不是 Qwen 结果。
- Coat：真实训练/随机评分数据，但使用既有开发面板，不接触保留测试集。
- AgentCF：作者 CDs 交互数据、8 位训练用户、每用户 2 个事件；共享训练和预决策排名冻结一次。Memory 与 Popularity 对照复用同一状态，二次使用明确记为*开发数据重用*。
- 所有实验：两个预设开发面板，注册动作、独立数值评价、已知目标 NDCG 辅助检查、未见标签不得进入 Qwen 方案上下文。

## 三组真实 Qwen 科研对照

| 科研任务 | 共享训练模型请求 | Linear 模型请求/实际实验 | Tree 模型请求/实际实验 | 结论 |
|---|---:|---:|---:|---|
| Coat Utility (第二次、Critic 任务协议修复后) | 0 | 3 / 0 | 7 / 1 | Tree 有可执行审计，但 gamma 注册反证失败 |
| AgentCF Memory | 88 | 3 / 0 | 11 / 0 | 缺少两个面板都支持的其他用户记忆写入-读取链；正确弃权 |
| AgentCF Popularity（重用上一组冻结状态） | 0 **新增**（历史训练成本 88） | 24 / 1 | 24 / 1 | 两个策略实际完成排名干预，但事前数值预测均未通过 |

### Coat 重要科研语义更正

Coat Tree gamma_stress 的两个面板 scalar gamma_reference_margin 约为 **−0.000457** 和 **−0.000498**，故 Qwen 方向预测的单独阈值检查通过；但是冻结参考机制评价发现两面板最优 gamma 配置不稳定，**registered falsifier = false**。旧报告的 forecast_passes_both_panels=true 是软件评价语义缺口，不是独立科学证据。

修正后 evaluator 重新从已保存的 execution.json 与 frozen_plan.json 计算审计，结果 **registered_action_falsifier_passes=false、forecast_passes_both_panels=false**。原始实验产物未改写。

- 原始 Qwen 记录：runs/tree-linear-coat-qwen-real-20261020-02/
- 新衍生复审：runs/scientific-audit-reconciliation-20261020-01/coat_gamma_revised_audit.json
- 新衍生汇总：runs/scientific-audit-reconciliation-20261020-01/qwen_real_development_summary.json
- 第一次真实运行（Critic 将 AgentCF 排名要求错误施加给 Coat，二者未执行）：runs/tree-linear-coat-qwen-real-20261020-01/

### AgentCF 科学支持边界

真实共享训练产生 88 次 Qwen 调用，来源 model_usage 为 73,685 prompt tokens 与 12,148 completion tokens；冻结训练 SHA-256：

a8546a588cde31b3c7c5920502c62dc9c49c9a6152d9da2d639f3b1b4db962fa

Memory 的广泛及定向消融在至少一个预设面板缺少跨用户 Item Memory 路径，两个动作均返回 ABSTAIN_UNSUPPORTED_CROSS_USER_REPLICATION。这不能解释为“没有 Memory 机制”，只说明当前观测无法识别两面板可复现机制。

Popularity 任务重用这一训练状态，两个策略各执行一个合法排序干预并独立审计，开发预测均未通过。复用训练既节约 Qwen 计算，也**不能**把这个已研究状态充当新独立确认。

- 训练与 Memory：runs/tree-linear-agentcf-qwen-real-20261020-01/
- 复用与 Popularity：runs/tree-linear-agentcf-popularity-qwen-real-20261020-01/

## 验证有效的架构能力

1. 四科研角色、受控动作与独立 evaluator 能共同完成真实模型驱动的方案选择与排序实验。
2. Tree/Linear 共用同一预注册训练状态、输入指纹、模型设置及请求上限；实际 Token 和执行次数分开报告。相同上限不意味着实际成本相同。
3. Critic 拒绝/机制证据不足/无效方向/注册反证失败互不混淆。
4. Tree 保留未被执行的 frontier 和审计子节点；有效反证失败不会触发虚假的“发现”。
5. 审计研究记忆仅链接可校验的数字证据；Research Skill 仍需要真正未见任务的独立确认。
6. 没有安装 AutoDebias、POPPER 或 AI Scientist-v2 的依赖，也没有执行外部生成代码。理论与实现的映射见 SCIENTIFIC_RESEARCH_REFERENCES.md。

## 后续应如何做（优先科学而非增加装饰）

- 预先定义结构上有完整其他用户写入/读取支持的作者训练面板，并记录其潜在分布；不要因已见结果而事后挑选更高分的 seed。
- 对独立新用户/新内容开展预注册反证，必要时增加组规模和无变化重复次数；区分 Qwen 本身排序噪声与训练语义随机性。
- 执行 AutoDebias/IPS/DR 的等数据、强方法基线；对 Exposure 用可识别的已知倾向环境，不将 KuaiRand 未知支持集结果当因果结论。
- 对 Tree、Linear、强固定规则、Audited Memory、Shuffled Memory 在未见任务上比较真实总 Token/GPU 成本和独立确认发现，保留失败。
- 参数化微调只能在已证明有机制支持的任务上增加，不应为了产生更多搜索节点把不可识别实验重复十次。

**结论：这次的“软件行为 + 真实数据 + 本地 Qwen”开发期链路已经实际运行，科学发现仍为零，RSI 自我改进未经证实。**
