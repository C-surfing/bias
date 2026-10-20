# RSI 五类 Bias：持续 Debias 工作流与公平基线（2026-10-20）

## 目标

**让科研 Agent 持续提出并试验可识别的 Debias 方法，不是连续提出 Memory 假说**。现有代码已经保留唯一五类偏差 Registry、四科研 Agent、有限实验树、Director 和独立 Numeric Audit。本阶段没有建立第五个科研 Agent 或另一套评价框架，只在原有 `five_bias_development.py` 登记真实数值动作，在原有 `scientific_director.py` 增加 `science-debias` 的任务持续调度。

单轮操作：读取冻结实际开发数据 → Diagnosis 提出假设及竞争解释 → Designer 从该 Bias 的**剩余注册方法**中选择一个 → Critic 反证审查 → 独立 Numeric Audit 计算已被冻结的真实开发数值 → Reflection → 当前方法失败则下一轮尝试**未用过**的方法。

**重要：这个版本是对已运行真实数据方法结果的封闭、按需披露的科学选择工作流。数值来源是 CPU 已执行的冻结开发证据，并非每次重新拟合新的推荐模型，也不是一个未见任务的真正 RSI 泛化实验。模型在方法选择前无法看到该方法的数字结果。**

## 冻结的动作和成功验收

| Bias | 注册方法 | 开发期动作接受标准 | 独立确认资格 |
|---|---|---|---|
| Selection | 估计倾向反比权重 cap=5、20、100 | 开发 Brier 改善至少 0.002、NDCG 不低于基线 99%、Hit 最多下降 0.02 | **开发候选**，未获独立确认 |
| Exposure | 同样训练样本数的随机参考 CTR、完整随机训练 CTR | 同一 item/tab 共同支持集的随机开发 Brier 改善至少 0.01 | **随机参考预测改善**，真实日志倾向概率缺失，不能声称 causal OPE |
| Popularity | 频次惩罚 0.05、0.10、0.20 | 头部曝光下降至少 0.003，NDCG 和 Tail Recall 各保留至少基线 99% | **去热门代理效果**，不识别不合理因果热门偏差 |
| Filter Bubble | 暂不执行臆造干预 | 原真实日志配平后主题熵方向反转；缺匹配多轮反馈干预 | **弃权**，不是 Debias 成功 |
| Fairness | Coat 训练集群体残差校准 | 性别群体 Brier 差距下降至少 0.0005，且任何群体 Brier 恶化不超过 0.001 | 当前方法失败；公平准则仅为**预测误差 parity** |

Developer 固定基线既可以做方案比较，也可约束 Qwen 的解释。若一个任务有新证据合格，只作为开发候选，不可认为统计显著或因果改善。三类结果必须严格分开：

- `development_effect_passes`：方向、效应阈值及质量保护同时符合；可能只是 Exposure/Popularity 代理结果。
- `development_debias_candidate`：Selection 的合法随机参考方法开发候选；但仍无独立确认，不能算科学发现。
- `science_goal_met`：在本开发实验中**永远是 false**；必须另有预注册、未见数据的独立评价才能修改外部科研验收。

## 持续执行与停止

运行 `python -B -m rsi_researcher science-debias`。Director 的冻结协议含所有任务、合法动作、真实证据文件 SHA256、效果门槛、最大轮数、最大 Qwen 请求、不同 Bias 的可识别性约束。每轮独立审计保留在 `director_state.json`。按冻结任务队列循环，如果失败会重新排入下一轮，但同一任务动作**最多尝试一次**。禁止用未见确认集、随机挑选种子或者 LLM 自评来重复优化。

- `STOP_DEVELOPMENT_DEBIAS_CANDIDATE_PENDING_CONFIRMATION`：至少预设数量的 Debias **开发候选**通过，且五类问题都已检查/明确弃权；此时立即冻结结果，进入独立确认准备，**不是 RSI 科研成功**。
- `STOP_NO_FURTHER_REGISTERED_METHODS`：合法方法耗尽，无可继续问题；必须增加科学上有效的新方法/数据，而不是无限重试。
- `STOP_ROUND_CEILING`、`STOP_MODEL_REQUEST_BUDGET`：达到资源边界；未完成工作明确保留。
- `FAILED_CLOSED`：模型或协议违反数值可信契约。历史痕迹不重写。

参考命令（必须先有**自己授权且独立**的 Qwen vLLM 服务）：

```bash
cd /data/cjh/rsi/experiments/rsi-multiagent-20261017-chatgpt
python -B -m rsi_researcher science-debias \
 --policy qwen \
 --evidence runs/five-bias-unified-development-20261020-02/five_bias_development.json \
 --config configs/qwen_five_debias_local_20261020.json \
 --max-rounds 14 --max-requests 90 --min-development-debias-tasks 1 \
 --output runs/UNIQUE_QWEN_FIVE_BIAS_RUN
```

固定基线不需要 Qwen：

```bash
python -B -m rsi_researcher science-debias \
 --policy strong_rule \
 --evidence runs/five-bias-unified-development-20261020-02/five_bias_development.json \
 --max-rounds 14 --output runs/UNIQUE_FIXED_FIVE_BIAS_RUN
```

## 固定规则基线实际结果

`runs/five-bias-fixed-debias-20261020-01/`：5 轮、0 次 Qwen 请求。
- Selection cap=5：Brier 改善 0.000122，未通过效果阈值。
- Exposure 等量随机反馈：共同支持集 Brier 改善 0.081114，属于**预测分布代理效果**，不算因果 Exposure Bias 解决。
- Popularity λ=0.05：Head 20% 曝光份额下降 0.002069，低于预注册 0.003；未通过。
- Fairness 群体校准：群体 Brier 差距恶化约 0.00185，明确失败。
- Selection cap=20：Brier 改善 0.003219，NDCG/Hit 的开发质量门槛通过；首次产生合格的 **Selection 开发候选**，停止并等待独立确认。

## 剩余科学挑战

- **必须跨真正未见任务比较 RSI 的优势**：现有 Qwen vs StrongRule 只在同一冻结开发动作表中选择方法，不能证明 LLM 能自主发明新 Debias 方法，也不能证明 Tree 或长期 Memory 的泛化收益。
- **统计保证**：预注册新面板、多指标和多方法比较的校正、成本、负结果；在独立数据上确认后才可声称发现。
- **提高真正五类覆盖**：Exposure 缺完整 logging probability；Popularity 缺需求/反事实相关性；Filter Bubble 缺反馈闭环；Fairness 缺质量约束下成立的改善。无法识别时退出是一项正确科学能力，但不是已经解决。
- **模型调用成本**：Qwen 训练/实验与固定规则要同口径计算 GPU/Token；模型服务维护独立，不干扰共享资源。

**验收原则：达到有证据的开发效果就停下来交独立确认；没有效果就在合法方法空间继续研究；空间或预算穷尽就报告科学阻碍。循环次数不是成功指标。**


## 2026-10-20 本地真实 Qwen3-8B 结果与证据等级

全部 Qwen 调用都发送至本会话独立的本地 vLLM（GPU 5，127.0.0.1:18427）；采用 schema-constrained JSON 的原 Diagnosis / Designer / Critic / Reflection，并复用同一份已冻结真实数据数值结果。真实推理费用与本地方法选择并非同一种成本，不能因为轮数更少就推断 Qwen 更经济。

| 方案 | Qwen 请求 | 轮数 | Selection 开发候选 | Exposure 结果 | Popularity 结果 | Fairness |
|---|---:|---:|---|---|---|---|
| \`strong_rule\` | 0 | 5 | cap=20，Brier gain +0.003219；先测试 cap=5 未达标 | 随机等量训练的预测代理改善 +0.081114 | λ=.05 未通过头部暴露改善阈值 | 失败，gap gain −0.001850 |
| \`qwen\` 首次协议 | 16 | 4 | cap=20，+0.003219 | 随机全量训练预测代理 +0.082316 | 模型误把 raw-head-decrease 写作 improvement-decrease，设计审查拒绝 | 同样的符号错误，被审查拒绝 |
| \`qwen\` 明确 gain 符号协议后的调试 | 16 | 4 | cap=100，+0.003735 | Critic 拒绝 | λ=.10，头部曝光减少 +0.005517，质量 guard 通过，**仍是代理指标** | 执行并反证失败，gap gain −0.001850 |

原始运行位置：
- \`runs/five-bias-fixed-debias-20261020-01/\`
- \`runs/five-bias-qwen-debias-20261020-01/\`
- \`runs/five-bias-qwen-debias-20261020-02-sign-contract/\`

两次 Qwen 不是盲法独立重复：第二次看到第一次的协议 bug 后修订 Prompt，使用同一开发数据。因此不能把第二次新选的 cap=100 和 popularity λ=.10 当作独立发现，不可用这两批数据估计泛化 RSI 增益。

**核心结论**：真实模型能够从尚未披露数字的合法 Debias 动作中选择候选，Critic 与 Python 评估会拒绝无效方法，Director 跨任务运行且仅在达到开发期阈值后停止。它没有实现新的算法公式自主发明或未见数据独立确认。固定规则也成功并且没有消耗 Qwen 请求，因此当前尚无 RSI 的增量科研效率证据。

下一阶段的正确评估是：把选出的策略及数值接受门槛冻结为真正独立确认协议，且预先补充 Exposure logging/positivity、Popularity 相关性参考、Filter Bubble 匹配多轮试验与 Fairness 有效且规范性的目标定义。不能通过继续在同一张开发结果表上试错来声称独立科学成功。
