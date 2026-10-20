# 2026-10-20 RSI 连续科研飞轮：科学验收协议与未完成项

## 1. 为什么历史版本只跑一轮

- `scientific_agents.REFLECTION` 内 `suggested_next_task` 的 enum 只有 `stop`。Reflection 的工作是总结一次经过独立评价的实验、提出仍然存在的竞争解释，不能单独授权下一轮。
- `scientific_loop.campaign` 只有有限 worklist；`scientific_tree.search` 自身也只有一棵树的深度、节点、请求与动作去重边界。
- 先前 Tree/Linear 对照实际只覆盖 1–2 个节点；当 Critic 拒绝、跨用户 Item Memory 支持不足或 Frontier 耗尽，研究就终止。
- 这些是**调度器/证据缺口**，而不是 Qwen 权重本身缺乏「持续思考」能力。修改 Reflection 的 enum 或添加无限 while 循环都不解决可识别性、科学评价和预算所有权。

## 2. 实现：Outer Research Director + Inner Scientific Tree

唯一新增的**调度责任**为 `rsi_researcher/research/scientific_director.py`。
不新增第五个 LLM Agent，也不复制数据执行器、独立评价器或 Research Memory 数据库。

外循环由 Python 控制：
1. 在第一次模型调用前冻结任务列表、固定种子计划、合法任务注册、总请求/Token/时间上限，以及科学门槛。
2. 根据上一轮已完成的数字审计和可识别性结论从冻结候选集中选择下一问题；Memory 机制证据不充分时，优先转向**同一输入、不同机制**的 Popularity 任务，不反复搜寻正向种子。
3. 调用原 `scientific_tree.search`：Diagnosis → 多候选 Designer → Critic → Python 执行/反证 → Reflection → 候选 Skill 记录；树级 Parent 与共享 User/Item Memory 来源仍由既有模块管理。
4. 对已完成节点保存 SHA256 可追踪 numeric audit，并将其作为下一轮同类任务的**开发期** Research Memory；不把 Agent 自述、未执行计划或验证标签混入经验。
5. 计算开发期证据是否有资格进入独立确认；否则在预算不足、连续无可用证据、任务计划耗尽、不可识别或执行异常时退出。输出 `director_state.json`、各 `cycles/cycle-xxx-<task>/` 的冻结证据与调用日志。
6. 从未提供一个允许 Qwen 修改代码、注册指标、隐藏测试标签、强制创建进程或自称验证成功的接口。

## 3. 优先可识别性：两级检验

- **pretraining structural support**：利用正训练事件中每个物品的写入者和无标签候选，只检查 A、B 两个面板是否存在 *可能* 的其他用户写入-读取链。任何面板没有潜在链，完全不启动 Qwen 训练/排名，不给研究 Agent 生成没有意义的计划。
- **posttraining changed-memory support**：即使潜在链存在，也必须观察训练后 Item Memory 实际改变且 A/B 均有跨用户读取机会，才能生成共享记忆机制实验候选。这是实际 Qwen AgentCF Memory 先前 `A=0,B=3` 所没有通过的门槛。
- 正面通过这两级依然不代表因果偏差、真实负反馈、统计显著性或独立确认。Selection/Exposure/Popularity 各有不同的识别条件，不共享一个万能门槛。
- 训练状态的 SHA256 与输入 cohort/frozen ranking SHA256 必须全部校验；同 cohort 的 Memory / Popularity 可以共享原训练成本，但不是两份独立实验样本。

## 4. 明确、分层的验收与停止条件

**Level A：软件行为**
- 完整协议、节点冻结、Research Memory 来源、预算、跨数据环境隔离、未知支持弃权、异常失败封存、旧实验不改写；
- 完整回归通过。只能说明工程协议实现可靠，不代表科研假设正确。

**Level B：开发期准备进入独立验证（本次可自动达到的最高科学门槛）**
- 至少 **2 个不同的科研任务**有新的数值审计；
- 覆盖至少 **2 个不同任务家族**（Coat 与 AgentCF）；
- 每项均符合事前冻结的效应方向/量级阈值，**注册机制反证通过**，已知目标质量保护通过；没有把历史回放当新增观察；
- 任一条件不达标，即使某次 NDCG 上升也**不能**作为合格候选；
- 达标只输出 `STOP_DEVELOPMENT_READY_FOR_INDEPENDENT_CONFIRMATION`，不是 `SCIENTIFIC_SUCCESS`。

**Level C：科研/RSI 真正验收（当前系统尚未具备充分数据和程序，绝不自动冒领）**
- 未见的、事先注册的独立数据/用户/机制任务，必要的随机化和曝光可识别性；
- 与强固定规则、no-history、shuffled-history、可实施的 IPS/DR/AutoDebias 比较；总 GPU/Token/时间成本计入共享训练，不能只按节点数；
- 完成严谨统计检验、重复试验和多重比较控制；预注册 stop/alpha 预算，审计模型输出不可改阈值；
- Research Skill 的版本、候选来源和反例完整保留，独立复核后才能外部批准/promote；
- 若达标可产生**外部独立确认的科研成功**，不应由这次开发程序仅凭 LLM 文本设立 `scientific_success=true`。

**安全停止**
- `STOP_RESOURCE_BUDGET`、`STOP_CYCLE_CEILING`、`STOP_NO_IDENTIFIABLE_PROGRESS`、`STOP_SCHEDULE_EXHAUSTED`：预算/证据/任务已耗尽。不是科学成功。
- `FAILED_CLOSED`：发生执行/协议异常，保留文件并停止，不在半完成的节点上继续。
- `ABSTAIN_PREDECISION_MECHANISM_SUPPORT`：单任务不满足结构或已训练机制支持。调度器优先尝试不同机制，不随机挑选最佳种子。

## 5. 参考实现与取舍

- **AI Scientist-v2**：https://github.com/SakanaAI/AI-Scientist-v2 的 Experiment Manager、Budgeted Best-First Tree Search 和多分支运行记录启发了树内层；项目自己承认其更开放的搜索未必胜过清晰模板。不能将其模型生成代码的执行模式部署到共享服务器；其 Source Code License 应单独核对。
- **POPPER**：https://github.com/snap-stanford/POPPER 的主动证伪和序贯检验强调**控制 I 类错误**。本项目仍只有开发期重复面板与门槛，因此没有借 POPPER 之名声称统计保证。
- **DiscoPER**：https://arxiv.org/abs/2607.01131 强调 periodic second-order meta-reflection、识别以前探索中的遗漏/混杂。当前 Research Memory 只承载有来源的审计，director 仅有简单的支持失败重定向；尚无真正的周期性跨任务 Meta-Reflection 和经盲测验证的研究策略升级。
- **AutoDebias**：https://arxiv.org/abs/2105.04170 是数学优化自动去偏强基线。其官方实现依赖需要隔离，不能把旧库依赖安装进共用 Python 环境。
- **Darwin Gödel Machine**：https://arxiv.org/abs/2505.22954 的候选版本与验证思想可迁移到 Research Skill；禁止把开放式自修改代码和对未见任务结果自我认证直接接入科研控制层。

## 6. 尚未解决的关键科研工作（按优先级）

1. **Mechanism identifiability**：先补全 AgentCF 已知反馈、Item Memory writer-reader 因果对照，再区分 rank noise 与 stochastic training；不能仅通过热门曝光差值识别真实 Popularity Bias。
2. **独立盲测与 Type-I error control**：研发用 Coat 随机评分参照；预注册独立确认时才使用保留集；KuaiRand 若 positivity/exposure support 不足则退出而非强行评价。研究设计可考虑 POPPER 的测试思想，但不可直接套用误差界。
3. **强基线**：IPS/DR/AutoDebias/固定选择/Tree/Linear 真实同数据、同输入状态和匹配计算预算的对照。当前共用请求上限但实际 Token 消耗不完全相等。
4. **RSI 研究策略的可迁移自我改进**：Research Skill 仍是 unverified_candidate，不满足独立验证就不能升级；未来做真正的跨任务/跨样本 Meta-Reflection 而不是总结日志。
5. **有根据的参数搜索**：在具备识别条件之后探索注册干预的受控参数空间，保持指标、独立审计、随机种子对照固定；不要为了多生成节点而变更科学问题。
6. **生产级连续运行**：增加中断后可审计的恢复、人工暂停/终止、跨运行的来源验证与成本账目，不能依赖一个永久进程或无限 while。
7. **论文级研究设计**：预注册对照与统计功效、报告负结果、计算开销、不变性检验，并实证回答 Tree/Memory/Skill 相比 AutoDebias 等基线的额外科学价值。

## 7. 使用说明

`science-cycle` 只调用**已经运行的**本地 Qwen vLLM，不自行启动 GPU 服务：
```bash
cd /data/cjh/rsi/experiments/rsi-multiagent-20261017-chatgpt
/data/cjh/rsi/envs/research/bin/python -B -m rsi_researcher science-cycle \
  --config configs/qwen_scientific_cycle_local_20261020.json \
  --tasks agentcf_memory coat_utility agentcf_popularity \
  --prepared-state runs/tree-linear-agentcf-qwen-real-20261020-01/shared_training/task-001-agentcf_memory/prepared_agentcf.json \
  --seed 20270001 --cohort-size 8 --max-cycles 3 \
  --max-nodes-per-cycle 1 --max-total-requests 65 \
  --max-stagnant-cycles 3 --min-ready 2 --min-families 2 \
  --output runs/UNIQUE_DIRECTOR_RUN
```
同一训练状态可跨开发机制任务复用，源指纹必须验证；模型服务的端口和 GPU 由**外部用户授权的独立服务**管理。不可直接在被别的会话占用的 GPU 上启动进程。

**要点：能持续调度不等于无限研究，也不等于自动产生科学发现。高质量 RSI 的成功标准是可证伪、可停止、可迁移、可独立确认。**
