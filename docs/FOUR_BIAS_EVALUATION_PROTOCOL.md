# RSI Debias 四类 Bias 的可执行评价协议

主线：Popularity、Candidate Position、Selection、Exposure。暂停 Fairness、Filter Bubble。

## 评价目标
| 研究任务 | 主目标 | 推荐质量 | 数据识别边界 |
|---|---|---|---|
| Popularity | PopREO@5 越低越好 | NDCG@5、HR@5 | 训练期流行度五组机会代理，不等于因果流行度影响 |
| Candidate Position | |CPB_NDCG@5| 越低越好 | NDCG@5、HR@5 | 同用户同候选的顺序配对诊断，非网页点击位置偏差 |
| Selection | Brier 越低越好 | NDCG@50、Recall@50 | KuaiRec 接近完整反馈上的人为已知 MNAR 机制，非真实自然 MNAR 确认 |
| Exposure | 已知 logger pscore 下 SNIPS 策略估计效用越高越好 | 估计点击效用、ESS、置信区间 | Open Bandit 仅识别特定静态策略 OPE；SNIPS 不是曝光偏差消除率 |

## 独立科学判定
偏差主目标、推荐效用保护、证据与统计支持、计算成本四栏单独记录。
开发搜索阶段即使通过门槛也不等于独立确认；没有确认的发现数必须记为 0，模型请求/发现数记为不可定义。
伪阳性风险包括：仅 ARP 降低却 PopREO 恶化；只靠降低高召回组命中让 PopREO 缩小；仅让所有用户组质量下降而降低公平差异；仅基于 SNIPS 提升就声称曝光偏差被消除。
对 Popularity 要报告每组正例及命中、clean parser 配对和 Bootstrap。
对 Position 要报告输入置换配对、Top5 overlap、clean 率和 bootstrap。
Sports 开发集的原始 |CPB| 约 0.0107，旧绝对改善门槛为 0.02，故该研究树本来就难以达到目标，应在新的冻结试验开始前增加基线效应大小和统计识别检查，不应该事后修改本轮门槛。

## 架构
四任务共用一个证据契约，复用现有 Diagnosis / Designer / Critic / Reflection、Qwen 模型调用配额、真实评估程序。
Amazon 使用原有 LLMRank science tree；Selection/Exposure 使用共享四 Agent 的受限任务适配器。后者当前选择已实现的 IPW 截断与静态 OPE 策略，属于有界原型实验，不等于自发发明新的连续优化算法。
原主线仍受其他会话并行编辑，本轮未覆盖它们的未提交改动。

## 复现命令
Selection:
  PYTHONPATH=. python -B -m rsi_researcher.research.four_bias_workflow --mode selection --seed 20261029 --output runs/NEW_DIR
Exposure:
  PYTHONPATH=. python -B -m rsi_researcher.research.four_bias_workflow --mode exposure --seed 20261029 --output runs/NEW_DIR
Amazon Popularity:
  PYTHONPATH=. python -B -m rsi_researcher.research.four_bias_workflow --mode amazon --dataset Beauty --target popularity --users 200 --seed 20261022 --requests 1240 --output runs/NEW_DIR
Amazon Position:
  PYTHONPATH=. python -B -m rsi_researcher.research.four_bias_workflow --mode amazon --dataset Sports --target candidate_position --users 80 --seed 20261027 --output runs/NEW_DIR
随机搜索后处理:
  PYTHONPATH=. python -B -m rsi_researcher.research.four_bias_random_control --run runs/AMAZON_RUN --data datasets/amazon/processed/Sports --dataset Sports --seed 20261027
辅助量审查:
  PYTHONPATH=. python -B -m rsi_researcher.research.four_bias_auxiliary_metrics --dataset Sports --data datasets/amazon/processed/Sports --seed 20261027 --run runs/AMAZON_RUN
汇总:
  PYTHONPATH=. python -B -m rsi_researcher.research.four_bias_reporting --output runs/four-bias-unified-workflow-20261022-v1

## 同预算与跨域边界
当前 Fixed/Random 主要复用已冻结 Qwen 排名，Agent 产生科学家与新排名推理请求，实际请求成本不相同；不能称严格同算力优胜。
KuaiRec Position、KuaiRand 随机曝光、Open Bandit OPE 使用不同推荐环境或估计对象，不能将数值直接视为同算法跨数据集复现。
后续应统一任务特异的连续干预参数接口、共享独立评价、冻结开发/留出、请求预算和有效发现准入。


## 2026-10-22: Frozen multi-round Selection / Exposure scientific tree

新的统一 CLI：python -m rsi_researcher four-bias-tree

Amazon Popularity/Position 仍沿用原 LLMRank science.search；Selection/Exposure 共用原 Qwen Diagnosis/Designer/Critic/Reflection 与数值审计，但拥有各自可辨识的数据适配器，以及父子节点选择、失败类型驱动扩展、反证修订、跨轮 Development 记忆、去重、冻结和 Holdout 准入。当前不同任务的 frontier 逻辑还未抽出一个跨任务的单一基础类；不可宣传为完全同一个搜索器源码。

Selection：KuaiRec small_matrix，稳定用户哈希隔离 Train / Development / Confirmation (70/20/10)；本轮分别为 3,251,569 / 1,020,633 / 404,368 条真实反馈，确认用户 122。三种子只改变训练端已知 MNAR 额外缺失随机实现，而非独立收集三份用户数据。Designer 连续生成 Blend (0..1) 与 IPW Clip (1..5) 参数，固定对照为完整 Oracle IPW；主要结果是独立参考用户上的 Brier 和受约束 NDCG@50。

Exposure：官方完整 OBD 412,931,917 bytes ZIP，通过校验；Random/all 已排除之前观察过的示例前 10,000 行，实际 Development 1,091,462、Confirmation 272,865 条记录。三个 Seed 对应三个互斥时序区块组，不是独立网站日志。Random logger 的 action propensity 必须等于 1/80；BTS 独立示例日志构建商品频率排序。Designer 生成 top（5..40）和 mass（0.25..1）策略参数；SNIPS、IPS、ESS 和区块配对 Bootstrap 用于评价 **静态离线策略点击价值**，不能冒称为曝光偏差的因果消除。

冻结与确认：跨轮 prior 必须匹配任务、数据 SHA-256、三个开发种子和分割；已经打开 Holdout 的 Campaign 不能当成下一轮搜索祖先。冻结文件必须在 Holdout 读取前持久化。开发集未通过全部三种子 CI / utility / ESS 条件时不能打开确认。严格控制评价器是最终证据权威，任何 Agent 的 Reflection 只是猜想。

运行环境：NumPy 2.2.6 安装在本轮专属 runs/four_bias_numpy，不改共享科研虚拟环境。以下所有命令在项目根目录运行。

第一轮：
PYTHONPATH=.:runs/four_bias_numpy /data/cjh/rsi/envs/research/bin/python -B -m rsi_researcher four-bias-tree --task selection --nodes 2 --random-trials 2 --development-only --output runs/NEW_R1

第二轮：
PYTHONPATH=.:runs/four_bias_numpy /data/cjh/rsi/envs/research/bin/python -B -m rsi_researcher four-bias-tree --task selection --nodes 2 --random-trials 2 --prior runs/NEW_R1 --output runs/NEW_R2

Exposure 把 --task selection 换成 --task exposure。

本次证据目录：runs/selection-rsi-tree-20261022-round1、runs/selection-rsi-tree-20261022-round2-confirm、runs/exposure-rsi-tree-20261022-round1、runs/exposure-rsi-tree-20261022-round2-confirm。统一报告位于 runs/four-bias-scientific-tree-20261022-results/RESEARCH_REPORT.md，结构化证据位于同目录 SUMMARY.json。

限制：Selection 确认针对额外已知人工 MNAR，而非未知自然机制；Exposure 确认特定静态策略在随机 logger 下的点击价值，而非 Exposure Bias 消除。原 KuaiRec 之前存在全数据探索，不能称每个原始用户在整个项目历史中完全未见。本轮 Agent 并未在确认主指标上超过 Fixed；跨数据集同策略复现没有完成；Fixed/Random 的 CPU 策略评估与 Agent 的实际 Qwen 科研请求不等成本。

技术恢复：Exposure 确认首次计算完但保存时遇到 numpy.int64 JSON 序列化失败，之后不再提案或调整冻结策略，零 Qwen 调用重算同一方案。异常及严格恢复原因单独保存在 runs/exposure-rsi-tree-20261022-round2-confirm/CONFIRMATION_RECOVERY_AUDIT.json。
