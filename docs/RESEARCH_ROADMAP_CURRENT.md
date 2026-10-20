# RSI × Debias 下一阶段科研路线（2026-10-20）

**已完成工程基础**：五类 Bias Registry、Coat/KuaiRand/AgentCF 与合成世界实验任务、真实 Qwen3-8B 的 Diagnosis/Designer/Critic/Reflection、合法动作执行、独立审计、证据存储、预算型树搜索与 Outer Director、历史失败复盘。软件能执行科学研究，不代表研究方法已证明有效。

**已知负结论必须保留**：Coat 预注册 Selection 保留集联合质量门槛失败；真实 KuaiRand Exposure 无可靠部署 OPE；Popularity 不可仅以 head-share 下降宣称去偏；两期 Filter Bubble 趋势随样本量配平变号；旧 Fairness 群体截距使两组变差；真实 Qwen 组合设计未通过 Hit 质量约束；固定规则目前优于 Qwen 的开发发现/调用成本。

## P0：冻结可复核的现阶段证据与入口
- 以 `main` 当前的 `scientific_*`、`five_bias_development`、`bias_registry` 为标准研发路径；README 和架构只描述一套现状。
- 开发与确认数据绝对分离，保留单次 Coat 保留集已被使用的事实；所有原始实验文件只追加、不可覆盖。
- 旧 ExperimentJournal/ResearchMemory/历史演示只用于旧研究复核；不继续扩展其协议。

## P1：有识别前提的新科学任务与新独立数据
- 独立新 Coat 用户或新可用偏差基准；新 confirmation 分区需由未接触样本预注册，避免利用已开封保留集优化。
- KuaiRand 若缺已知 logging propensity 和目标策略 positivity，仅做预测迁移/支持集审计；不可强行标 OPE。
- AgentCF 预先选具跨用户 Item Memory writer-reader 支持的新训练面板，固定重复测量，区分真实干预效应与模型噪声。
- 真实纵向 Filter Bubble 需同初态多轮 matched policy 干预；无则明确弃权。

## P2：可信的强基线与成本公平
- 等数据、等划分、严格复现 IPS/SNIPS/DR 与 **AutoDebias 官方实现**；隔离依赖，不污染服务器共用 Python。
- 比较 strong fixed rule、no-history、audited-memory、shuffled-history、Linear、Tree/Director 的相同总预算，报告真实 GPU 秒/Token/实验次数。
- 对失败实验保留 denominator，不只汇报“最好的找到的那一次”。

## P3：统计性证伪与可确认科学结果
- 明确零假设、最小重要效应、统计功效、独立确认单位、置信区间、多重检验校正及序贯停机规则。
- 借 POPPER 的主动反证思想，不能直接宣称已实现其 e-value/I 类错误保证；对独立可识别任务设置审计。
- 任何验证标签都不得回流到 Designer 的后续开发选择；失败时换新独立任务，不复用保留集。

## P4：RSI 真正自我改进
- 对候选 Research Skill 先进行跨任务、未见机制的盲测；只有外部批准才能提升可复用科研策略。
- 借 DiscoPER 的周期性二阶反思与 AI Scientist-v2 的树搜索，改进研究假设与实验选择的样本效率，而非无限生成候选。
- 报告主要终点：**每单位真实计算成本的独立确认去偏发现数**，辅助统计：拒绝/弃权、失败率、效用损害、机制覆盖和证据可追溯性。

## 不做的事
不重造五套 Agent，不让 Qwen 自行修改评分程序，不把模拟数据说成真实因果结果，不为反复验证而增加无价值框架，不删除真实科学实验记录，不在已开封 Coat holdout 上“试到过关”。
