# Coat + AgentCF：Qwen 科研假设与反证 Pilot（2026-10-17）

## 核心研究问题
Qwen 是否能在相同受控证据下，主动产生有竞争解释的假设、选一个可执行反证，
再根据真实干预结果改进研究决策？真实科学发现数、开发信号与成本必须分开。

本次 **不是独立 RSI 确认实验**：Coat 随机评分开发集已经被前轮研究使用，
只在开发集用户中划分 A/B 两组；AgentCF 复用过去 20 用户训练状态中的
10 个开发查询（A/B 各 5 人），不是不同训练种子或独立用户物品状态。

## 完成的实验
- 任务一 Coat/Selection 估计倾向鲁棒性：gamma 扰动 vs 随机错配用户倾向负对照。
- 任务二 Coat 校准与效用：同样的训练参数按两组开发用户比较 Brier、NDCG@5 与长尾召回。
- 任务三 AgentCF Item Memory：从冻结状态恢复候选物品初始记忆，冻结用户记忆与候选不变，重新调用本地 Qwen 排序。
- 任务四 AgentCF Popularity：保持冻结用户和物品记忆，仅移除流行度声誉提示，重新调用同一个 Qwen 排序。
- 两种 AgentCF 消融均使用相同来源的原始排序和同条件重复对照，避免把模型自身随机性直接认作机制。

所有方案都在真实干预之前提出并锁定；所有结果均由 Python 执行器计算，
Qwen 仅负责规划和复盘，不具有修改评价器、独立测试数据、任意 Shell 的权限。

## 严格研究发现统计

| 组别 | 任务完成 | 符合预登记门槛的开发信号 | 独立确认的有效科学发现 | 规划+复盘 Qwen 调用 |
|---|---:|---:|---:|---:|
| 强固定规则 | 4 | 1 | 0 | 0 |
| Qwen 无历史 | 4 | 2 | 0 | 8 |
| Qwen 相关历史提示 | 4 | 2 | 0 | 8 |

**这张表的 1/2 不是“真正发现数”。**
它只是选择的具体反证动作在这两个已使用过的开发面板上通过预设数字检查。
Qwen 假设本身仍是自由文本，没有被事先转成指标/效应方向/最小幅度的正式预测合同，
所以“被选择的实验有信息”不能等同“生成的科学假设为真”。

两个 Qwen 组在四个任务的实验动作完全相同；相关历史只是输入几条已有实验结论文字，
**本轮尚未验证结构化 ResearchMemory 检索和策略修订的跨任务收益**。

## 实际测量要点

Coat：
- 倾向负对照（cap100）：两组原始估计倾向对应 Brier 为 0.15105 / 0.14804，
  两种错误用户错配的最佳 Brier 为 0.15731 / 0.15323；
  错配不是有效的因果倾向，实验仅说明当前估计权重的映射影响开发评分。
- 截断100 对未加权：两组 Brier delta = -0.003297 / -0.004175；
  NDCG5 ratio = 1.00098 / 1.00197（符合 >=0.99 质量门槛）；
  **长尾相关召回 delta = -0.01687 / -0.05628**。
  因此不能称“完全不损害效用”。
- gamma 0.75/1/1.25 的最优权重在两组用户上不一致，严格稳定性未通过。

AgentCF：
- 同条件精确排名一致性 A/B = 0.8 / 1.0。每组仅 5 用户，不是对随机性充分建模。
- Item Memory 恢复到初始记忆使完整排序分别在 100% / 80% 查询中改变，
  但 NDCG@5 从 0.4123->0.3897、0.2774->0.2262，两个分组质量门槛都失败。
  说明 Item Memory 会影响排序的开发观察，但不能据此认定改善。
- 流行度提示干预：热门超额曝光变动 0.00 / -0.04，未在两组稳定降低；
  NDCG@5 两组有所改善，但指标和样本量不足以建立经典 Popularity Bias 的因果结论。

## 模型与实验成本（已实测）

- GPU: Qwen3-8B, 单独使用 GPU 5，端口 18329，专属进程组。
- Qwen 新请求总计 **60 次**，其中**8 次规划 + 8 次复盘 + 44 次 AgentCF 排序和必要的格式修复**；
  排序结果被不同规划组共享，并没有为每组再重复产生不同训练状态。
- 真实 API 用量：输入 **51,292 tokens**，输出 **5,303 tokens**，总计 **56,595 tokens**。
- GPU 服务与结果阶段总耗时约 105 秒（包括启动和释放）；GPU 5 最后 14 MiB。
- Coat 多项干预在 CPU 上运行，三种动作总计算耗时低于数秒。
- 研究规划组各只比较 4 个计划选择和 4 个反证复盘请求，8 次 Qwen 调用；
  其它 44 次排名请求属于共享执行成本，不能简单同时记到每一组。
- 因为动作空间、预算和执行调用复用规则与上一次研究不同，此结果不能与
  上轮 14 个 OPE 动作选择问题直接合并为一项总体 RSI 指标。

## 科研反证与 Research Memory

独立程序识别至少 **两处 Qwen 复盘与观测事实相矛盾**：
- 无历史组宣称排序质量下降，但此实验两组 NDCG@5 并未下降；
- 相关历史组称长尾未受影响，但两组 long-tail relevant recall 均下降。

所以 Qwen 自然语言“lesson”不能直接升级成高置信度 Research Memory。
新生成 SOURCE_CHECKED_RESEARCH_MEMORY.json 内含 5 张去重的源文件 SHA 验证卡片，
含实验通过与不通过的记录，**0 张已确认，0 条原始 Qwen 主张自动入库**。

## 项目文件位置
- 来源唯一主工程：/data/cjh/rsi/project/rsi_researcher/
- 四项研究任务和冻结规则：research/crossbias_discovery.py
- 受控 Qwen Agent + Rank 执行：research/crossbias_runner.py
- 数值一致性及费用审计：research/crossbias_audit.py
- 只用事实生成 Research Memory：research/crossbias_memory.py
- 全部实际实验原文与文件指纹：runs/crossbias-discovery-pilot-20261017/
  包含 protocol.json/protocol.sha256、plans.json、executions.json、
  assessments.json、reflections.json、calls.jsonl、comparison.json、
  INDEPENDENT_AUDIT.md、SOURCE_CHECKED_RESEARCH_MEMORY.json、vllm.log。
- 原始 Qwen 和旧冻结项目不变；没有接触任何真实确认集。

## 下阶段：不要继续刷这四道题
1. 先建立**结构化预测契约**：每次 proposal 必须预登记可计算
   target_metric、effect_direction、minimum_effect、confounding_alternative、
   rejection_if、expected_quality_cost，而不只是自然语言 hypothesis。
2. 强化确认数据与训练状态隔离，AgentCF 做多个新种子/用户-物品内存状态，
   使用重复排序区分采样噪声，并用独立标注解释错误记忆。
3. 用新的、未见研究任务验证是否能获得真实的“可复核发现/计算成本”，
   而不使用之前实验的开发证据重新挑选提示词或改变成功阈值。
4. 若未达到标准必须报告 0 发现，不能把研究动作合理、数值改善或 Qwen 自评当成 RSI。
