# 五类 Bias：预注册确认、可识别性反证与持续 RSI 真实实验（2026-10-20）

## 研究状态与版本

唯一科学问题分类：`rsi_researcher/research/bias_registry.py` 中的 Selection、Exposure、Popularity、Filter Bubble、Fairness。Qwen 四科研 Agent、既有 Director、评价器与真实数据环境不新增角色或另一套逻辑。

证据链：
- `runs/coat-fivebias-selection-debias-20261020-01`：未观察保留集前的开发方法对照。
- `configs/selection_confirm_protocol_20261020.json`：在访问保留标签之前以提交 `6be30d4` **冻结**的唯一确认方法（cap=100）、阈值与分割规则。
- `runs/coat-selection-preregistered-holdout-20261020-01`：对 2209 条未评分 Coat 随机评分的**单次冻结保留集确认**。
- `runs/five-bias-unified-development-20261020-04-normalized-dr`：五类真实/受控开发数据证据。之前 `03-paired-dr` 中模拟 head_capped 目标概率和为 2（错误），已明确作废；修正目标分布后以新路径运行，未覆盖旧输出。
- `runs/filter-bubble-paired-simulator-20261020-01`：完整两组不重合模拟种子与无反馈负对照。
- `runs/rsi-joint-design-fixed-20261020-01`、`runs/rsi-joint-design-qwen-20261020-01`：新组成公式的固定规则与真实 Qwen3-8B 提案。

## Selection：预注册保留集确实否证了质量约束

原开发中，cap=100 相对未加权 Brier 改善 0.003735，开发 NDCG 较稳定。事前冻结单独保留集判断条件：

1. Brier 改善至少 0.002。
2. 配对用户 Bootstrap 的（处理后减基线）Brier 改善区间上界小于零。
3. NDCG@5 不低于未加权基线 99%。
4. Hit@5 下降不超过 0.02。

保留集实际：Brier gain **0.0034348**；条件性用户 Bootstrap 区间 **[−0.006668, −0.003177]**；NDCG 保留比例 **0.980697**；Hit 下降 **0.005682**。**第 3 条失败；总体 `SINGLE_HOLDOUT_CRITERIA_FAILED`。** 必须将其作为研究失败。保留集是同一批已知用户的其他随机评分而非新用户，propensity 是作者估计，不是已知真实登录倾向；Bootstrap 未计整个训练样本和用户间 item 共享的依赖。不得按此失败再改 cap 或阈值并复用同一保留集宣称确认。

## Exposure：真实随机反馈与受控识别不能混用

真实 KuaiRand-Pure 上之前已经完成同训练量、共同 item×tab 支持集下，随机曝光训练 Brier 约 **0.1457**，标准曝光训练 **0.2268**。证明真实随机参考分布存在模型失配，不提供所有样本的已知 logging propensities 或可部署 target policy 的可识别 OPE。

本次 `exposure_synthetic.evaluate_dr` 在**完全受控且 logging distribution 已知**的模拟世界中，新增独立训练 nuisance 奖励模型与独立评价流。目标策略为不使用隐藏真实奖励的 uniform 与 normalized head-capped；比较原始 logged click、IPS、SNIPS、Doubly Robust 的 32 次重复 RMSE。**模拟器不能解决真实 KuaiRand 倾向缺失**。负结果同样保留：受控场景中 naive 估计有时比 DR RMSE 更低，并不证明 DR 普遍优越。

注意：修正以前的 `03-paired-dr` 版本无效，不能引用它关于 head_capped 的任何因果/数值结果；结果来自修正后的 `04-normalized-dr`。

## Popularity：相关性机会是首要反证

在 Coat 随机评分用户候选中，计算**已知相关（评分≥4）物品中头部物品所占比例**，作为有限机会参照。原始排序热门曝光份额比已知相关候选中的热门份额低约 **0.013846**；λ=.10 惩罚后差异扩大至约 **−0.019363**。

这意味着仅减小热门份额不等于减少不合理热门偏差。该分析受限于随机评分小候选集、同已知用户与未观察目录。不能从负差值推出整个系统不存在流行度偏差。

## Filter Bubble：同初态、多轮、匹配反事实

新增 `feedback_pair_control.py`。模拟器每位用户有固定、程序生成但不向策略公开的真实兴趣；基线、α=.22 均匀探索和无反馈负对照共用初态、50 轮潜在随机数及推荐更新规则。两组**不重合种子**各 32 × 60 = 1920 条用户轨迹。

模拟未见种子：归一化主题熵提升 **0.174780**（配对 Bootstrap CI **[0.170863,0.178926]**），潜在相关性期望点击率下降 **0.017689**；无反馈冻结历史与基线的熵差 **+0.120342**。预设熵提升≥0.08、效用损失≤0.025 与负对照等条件均通过。

这可验证**受控模拟世界的反馈闭环机制与干预效应**，但不能冒充 KuaiRand 上真实的多轮因果 Filter Bubble Debias。真实两时期数据仍坚持 `ABSTAIN_NO_MATCHED_INTERVENTION_TRAJECTORY`。

## Fairness：新方法（非之前失败的组截距）

原来的群体截距在 Coat 开发集令 men/women Brier 差距从 0.005924 扩大到 0.007775，失败。

新增 **训练得到的无权重/倾向加权 item 分数线性混合**（不是按群体强制重排），审计同一随机评分开发集的分组 Brier、总体 Brier 和各组不可恶化超过 0.001。25% 无权重混合时：群体 Brier 差距 **0.005924→0.004623**，总体 Brier **0.149546→0.148983**，通过事前开发期目标改善至少 0.0005 和分组效用保护。更大混合量有改善差距但不一定保护各组误差。

**严格作用域**：这只是 Coat binary men/women **预测误差 parity** 开发候选，不能泛化为 Equality of Opportunity、人口公平或真实社会性群体利益；更不能对这份已使用保留集的 Coat 重复试错并宣称独立确认。

## 超越固定参数选择：真实 Qwen 自拟组合实验

`science-debias --joint-design-only` 允许现有 Designer 通过结构化、带数值范围的研究调用，提出 `alpha∈[0,1]` 与 `beta∈[0,.15]`，组合**两种不同算子**：倾向加权训练分数与未加权分数混合；再对排序添加训练频次惩罚。Critic 仍须批准；Python 直接重新运行 Coat 开发分数评价，数值**不是**从预存的动作结果表查出。该能力是有限语法的**新实验设计**，不是任意 Python 代码自修改，也不是独立新用户发现。

- 固定组合 alpha=.75、beta=.05，开发 Brier 改善 **0.004298**，但 Hit 质量 guard 失败；合格开发候选为 0。
- 真正 Qwen3-8B 在独立 GPU 服务上提出 alpha=.70、beta=.10，开发 Brier 改善 **0.004290**，但 Hit@5 **0.931429**，相比原未加权 Hit@5 **0.954286**，下降超过冻结的 0.02；**质量 guard 失败**。合格开发候选为 0。共计 5 次 Qwen 请求，四科研角色均参与，设计额外调用用于提出参数。

两者对比表明**Qwen 能生成和执行新组合实验，但该实验在目标指标上没有成功，也没有显示优于固定规则**。不应因同一开发集的 Brier 较大改善而抹去 Hit 的失败。

## 持续循环、证据等级与科研成功

原有 `science-debias` 继续使用**五类真实开发证据**选择候选。它可以在一个方法失败后尝试不同方法，并对新 Fairness 混合候选实现独立分组效用约束。

重要分级：
1. 代码运行成功、Qwen 提出有效 JSON ≠ 去偏科学成功。
2. 真实开发分数改善或受控模拟干预改善 ≠ 未见真实数据独立确认。
3. Selection 的一次冻结保留集检验**没有通过联合验收**，所以独立确认去偏发现总数仍是 **0**。
4. 真正连续 RSI 应在**新数据、独立任务/环境与预注册强基线**下评价每次新设计，而不是在同一个已用过的 Coat 保留集上迭代直到正结果。
5. 改进目标是在未见任务上的**单位计算成本有效去偏发现数**，相对于强规则、no-history / audited-memory / shuffled-memory、IPS/DR/AutoDebias。当前还没有该跨任务 RSI 优越性证据。

不可识别或方法空间耗尽应安全退出并记录为何缺少证据；无上限等待“直到有好结果”会制造错误发现，违反当前科研协议。

## 原始来源与参考

- Schnabel et al. (ICML 2016), Recommendations as Treatments: https://proceedings.mlr.press/v48/schnabel16.html
- SakanaAI/AI-Scientist-v2: https://github.com/SakanaAI/AI-Scientist-v2 。借鉴有预算的树状实验与审计，不执行其生成代码；原仓库有运行生成代码的安全提示。


## Actual multi-task RSI continuation experiment

Frozen real-development input: \`runs/five-bias-unified-development-20261020-04-normalized-dr/five_bias_development.json\`.

| Arm | Completed rounds | Qwen requests | Identifiable scoped development candidates | Stopping state |
|---|---:|---:|---|---|
| strong_rule | 8 | 0 | Selection and groupwise predictive-error Fairness | STOP_DEVELOPMENT_DEBIAS_CANDIDATE_PENDING_CONFIRMATION |
| Qwen 3-8B | 7 | 28 | Selection alone | STOP_MODEL_REQUEST_BUDGET |

The Qwen session revisited unsuccessful methods across tasks; it did not terminate
on its first failed Critic. A full followup round is budgeted before starting;
at 28/32 calls it correctly declined to start a potentially 5-call composed
method. Qwen also observed a development-only Popularity proxy reduction, never
promoted to a second scientifically identified Debias discovery.

The fixed baseline satisfied two DEVELOPMENT gates earlier with zero model
requests; this is a negative finding for incremental RSI scientific efficiency
under the measured current protocol. It does not imply fixed rules universally
win: one task portfolio, one development evidence file, no matched heldout
generalization measurement, and no attempt to equalize actual Qwen/CPU costs.

Outputs:
- \`runs/five-bias-fixed-two-goals-20261020-01/director_state.json\`
- \`runs/five-bias-qwen-two-goals-20261020-01/director_state.json\`

**None of the above overrides the predeclared Coat holdout failure.** The
scientific success counter remains zero, with a real observed failure as an
auditable next-task input, not an excuse to reuse the same holdout.
