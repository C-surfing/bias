# RSI 当前架构和归档边界（2026-10-17）

## 代码入口与目录职责

项目唯一可编辑主包：`/data/cjh/rsi/project/rsi_researcher`。
冻结 AgentCF 发布版：`project/releases/20261016-five-bias-validation/`。
合并前原版备份：`project/releases/20261017-pre-consolidation/`。
独立真实研究历史：`experiments/classic-five-20261016`；
旧 Planner 实验：`outputs/research-planner-v2-20261016`、
`outputs/research-planner-integration-20261016`；全部保持原样，不直接继续扩张。
统一 CPU 原型历史：`experiments/unified-research-20261017-chatgpt`。

## 主包职责

- `research/bias_registry.py` + `configs/bias_tasks.json`：五类经典任务冻结注册表，
  Popularity / Exposure / Selection / Filter Bubble / Fairness；AgentCF Position/Conformity/Memory 只是机制诊断。
- `datasets/`：沿用作者数据来源，Coat 评分 0 是缺失、不作负例。
- `research/planner.py`：EvidenceRef / ResearchSituation / ResearchProposal / review 及可拒绝行动前提。
- `research/planner_adapters.py`、`planner_tools.py`：发展期真实证据与只读 CPU 诊断。
- `research/planner_integration.py`：冻结 AgentCF 设计审查及外部签名执行许可边界，
  DESIGN_ACCEPTED 不意味着 GPU 或独立确认可执行。
- `research/coat_selection.py` + `research/unified.py`：Coat 真实开发评分与
  固定规则/旧 Qwen 提案重放。支持最多 3 个外部提案，先由审查门槛检查，不自行调用模型。
- `research/kuairand_transfer.py`：KuaiRand Exposure 证据审计；
  不满足共同支持和记录策略识别前提时退出。
- `research/coat_sensitivity.py`：Coat 估计倾向在预声明 gamma 扰动下的稳健性。
- `research/memory.py` 和 `research/journal.py`：
  具有来源约束的开发记忆与旧 ExperimentJournal；持久结构的最终收敛仍待继续。
- `research/agentcf_loop.py`、`recommenders/agentcf.py`、`evaluation/`：
  复用 10 月 16 日已经验证的 AgentCF 反馈/记忆/干预和独立质量测量。

## 目前确实验证过的结论

Coat：训练 6,960 / 随机开发 2,065 / 保留随机确认 2,209。
原始估计倾向截断 100 使 Brier 相对未加权约降 2.44%，但 Hit、长尾召回有损。
预声明 gamma 0.75/1.0/1.25 的最佳 cap 分别 100/100/20；开发超参不稳健。
没有使用随机保留分区调参或评价，也无法以作者估计倾向当成已知处理赋值概率。

KuaiRand：普通/随机日志共同物品数 201、Jaccard 0.0767，缺确切同用户同物品跨策略匹配，
不构成可识别的因果去偏结果。

Qwen：旧决策动作先导六题无历史 4/6、有历史 3/6；新构造八题为 6/8 与 7/8；
参考答案不是外部金标准。Coat 固定网格与有历史 Qwen 均可选 100，
**尚无跨任务 RSI 增益证据**。

## 工程与安全

- 新 CPU 输出一律放在 project/runs/ 下不同的新目录。
- 冻结版本、归档 Qwen 原始回复、测试保留集、模型权重不被改写。
- 模型不具有任意 Shell 权限，不能自己改变评价器和科学成功标准。
- 对外报告必须区分「程序执行成功」、「指标有变化」与「独立证实科学效果」。
- 迁移记录详见 docs/CONSOLIDATION_MANIFEST_20261017.json。

## 新增受控曝光基准

- research/exposure_synthetic.py：已知正概率随机曝光、真实潜在偏好由合成机制定义。
- 在相同随机日志上计算两个目标策略的 IPS、SNIPS、Naive 和已知策略真值；独立随机种子重复。
- 对隐藏 propensity 的条件明确 ABSTAIN_NO_PROPENSITY，不输出伪 IPS。
- CLI: python -m rsi_researcher bias-exposure-sim --output runs/new-unique-name。
- 只用于离线估计量校准，不给 KuaiRand 日志添加不存在的真实倾向概率。
