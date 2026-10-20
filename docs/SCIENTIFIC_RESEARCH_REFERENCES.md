# RSI × Multi-Agent × Debias：论文代码参照与科学协议（2026-10-20）

## 研究目标和贡献边界

本项目不是再复现一个提升 NDCG 的模型，也不是让 LLM 以多轮 Prompt 自动挑选最优指标。本项目研究能否把**可识别的推荐偏差问题**转变为**可证伪的科学任务**，让科研代理提出有区别的解释、选择合法干预、接受独立测量、根据失败经验调整研究策略，并最终在**未见任务**上提高单位成本的独立确认发现数量。

所有 Agent 模型输出都是候选假设、实验计划或反思；不是实验真值。数值评估、数据划分、可执行动作和确认资格均由 Python 的受控实验协议决定。

## 关键论文及参考代码

### 1. 推荐系统偏差、因果评价和自动去偏

- **Bias and Debias in Recommender System: A Survey and Future Directions**, Chen 等，TOIS（https://arxiv.org/abs/2010.03240）。借鉴偏差分类和识别前提。经典 Bias Task Registry 是科学语义权威，研究任务协议再映射实际数据和合法操作，避免用「热门曝光」直接宣称已识别为因果偏差。
- **Recommendations as Treatments: Debiasing Learning and Evaluation**, Schnabel 等，ICML 2016（https://proceedings.mlr.press/v48/schnabel16.html）。明确推荐选择机制、随机参照、倾向权重和评价目标。Coat 的随机评分参考优先于普通有偏准确率；未经验证的倾向估计不等于真实曝光机制。
- **AutoDebias: Learning to Debias for Recommendation**, Chen 等，SIGIR 2021（https://github.com/DongHande/AutoDebias；https://arxiv.org/abs/2105.04170）。官方实现利用小规模均匀采样数据做元学习去偏。它是**自动参数学习**强基线，不等于具备机制假设生成能力的科研 Agent。不能直接以更强泛化或 NDCG 单指标认定 Agent 胜过 AutoDebias。其较旧的 PyTorch 依赖不得直接安装到共享服务器环境。

### 2. 主动证伪与实验树搜索

- **POPPER: Automated Hypothesis Validation with Agentic Sequential Falsifications**, Huang 等，ICML 2025，官方代码 https://github.com/snap-stanford/POPPER。提取「高层假设 → 可检验含义 → 反证实验 → 程序化证据」的职责边界。POPPER 的序贯检验涉及 I 类错误控制；**本项目当前的两面板效应阈值不是统计显著性检验，严禁称其具备相同的错误率保证**。后续确认实验必须事前预注册独立样本与合法序贯停机规则。
- **The AI Scientist-v2**, SakanaAI，https://github.com/SakanaAI/AI-Scientist-v2，重点参照 ai_scientist/treesearch/agent_manager.py、parallel_agent.py、bfts_config.yaml。提取预算管理、多个候选节点、分叉、失败路径保存、实验反馈、阶段式拓展。不复制通用代码生成/论文写作框架。原仓库提示会执行模型生成代码且要求受控沙盒，其许可证另有明确使用要求；本项目采用原创、注册动作驱动的调度器，不引入无沙盒任意代码执行。
- **Darwin Gödel Machine**, Zhang 等，arXiv 2025，官方代码 https://github.com/kew-lab/darwin-godel-machine。借鉴「保留竞争版本 → 独立基准验证 → 有资格再升级」的进化流程。目前研究策略 Skill 只能产生 unverifed_candidate；未经跨任务独立确认不得视为 RSI 进步，更不得直接替换可信执行器或评价器。

### 3. AgentCF 与推荐记忆反馈

- **AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems**, WWW 2024（https://arxiv.org/abs/2310.09233）。研究 User Memory 与共享 Item Memory 的写入/读取传播，固定候选、训练事件和评分协议，采用跨用户针对性消融。
- 用户未选择比较物品 ≠ 用户确实不喜欢：既要审查 User Reflection，也要审查 Item Update 的反馈解释。完整纠正属于联合语义干预，不能据此归因其中某一阶段。
- 作者 CDs 每个查询只有有限的已知验证正例，无法通过「未知候选没有点击」推断负偏好；小样本下 Item Memory 写入/读取链缺失，正确结果是科学弃权。

## 对应工程实现

| 科学责任 | 代码所有者 | 关键约束 |
|---|---|---|
| 偏差定义/可识别性 | bias_tasks 注册表与 scientific_experiments.task_spec | 历史冻结来源不修改 |
| 证据与假设 | scientific_loop.public_evidence、DiagnosisAgent | 不暴露验证目标标签 |
| 实验候选与预算 | scientific_tree.search、DesignerAgent | 保存父节点与未选候选；不得自改评价 |
| 审查 | CriticAgent、scientific_loop.review_plan | Coat 与 AgentCF 使用不同识别协议 |
| 冻结推荐状态 | scientific_experiments.train_and_observe_agentcf | 兄弟节点/跨策略共享相同 sha256 |
| 模型执行 | scientific_loop.episode + scientific_experiments.execute | 只有注册动作、无任意 shell |
| 审计 | scientific_experiments.independent_verdict | 数值预测、注册反证、效用、独立确认分别记录 |
| 科研记忆 | scientific_loop.verified_experience | 只读 SHA256 记录的开发数值审计，非反思文本 |
| 科研技能 | scientific_skills | 未见任务独立验证 + 外部批准；不能自封成功 |
| 搜索策略比较 | scientific_comparison.compare_tree_linear | 共同输入、硬预算、真实调用成本、无发现虚报 |

## 已有真实运行的含义

2026-10-20 Qwen3-8B 在单独 GPU 5 的 vLLM eager 服务上实际推理，模型参数与调用日志留存于各运行目录。Coat 科研搜索中，树提出了两个候选分支，Critic 审查后有一项进入数值评价；方向阈值通过**不等于**注册反证通过，更不等于真正去偏或独立发现。

AgentCF 作者 CDs 首批 8 用户共享训练发出了 88 次模型请求，跨策略复用了同一训练状态。跨用户 Item Memory 写入/读取链在一个预定面板缺失时正确弃权，不得反复挑选数据种子以制造阳性发现。相同冻结状态可用于另一任务的**开发期重用**，但不能冒充独立验证。

## 尚未完成/不得宣称完成的工作

1. **Stage 2**：合法参数空间的探索、搜索数据与独立选择数据分离，特别是 Coat 截断权重和定向记忆消融比例。当前主要是注册动作树搜索而非通用超参数搜索。
2. **POPPER 意义的统计确认**：两面板实用效应阈值没有 Type-I 错误保障。需独立样本、可检验的零假设、预设最小效应和多重检验控制。
3. **AutoDebias/DR 强方法对照**：应在隔离环境复现官方实现并匹配随机参照数据与训练成本，不能粗暴安装到共用 Python 环境。
4. **RSI 研究收益**：Tree vs Linear 同一预算上限只说明资源上限可比；实际 Token/GPU 成本通常不同。必须以未见任务、独立科学发现、实验/推理成本和负结果同时报告。
5. **树搜索安全性**：崩溃节点必须失败封存；生产完整恢复、跨多个数据域的大样本确认、搜索选择偏差控制尚需专门验证。不能因软件单测全过声称科研已完成。

原则：宁可报告机制支持不足、Critic 拒绝、效用下降和零发现，也不把回放、数值阈值或语言模型自述升级为新的科研贡献。
