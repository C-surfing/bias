# 五类推荐 Bias：实际 Debias 开发实验与识别边界（2026-10-20）

本报告以 **`bias_registry.py` / `configs/bias_tasks.json` 的五类任务**为唯一概念来源；不把 AgentCF Memory 当作第六类 Bias。代码只添加一个科学开发期数据适配器 `five_bias_development.py`，沿用原有 Coat 训练/随机评价、KuaiRand 随机日志及受控曝光模拟。没有新增科研 Agent 或评价框架。

运行（请勿覆盖旧证据）：

```bash
cd /data/cjh/rsi/experiments/rsi-multiagent-20261017-chatgpt
/data/cjh/rsi/envs/research/bin/python -B -m rsi_researcher bias-five \
  --output runs/UNIQUE_FIVE_BIAS_RUN
```

完整可机读产物：`runs/five-bias-unified-development-20261020-02/five_bias_development.json`。

## 1. Selection Bias — 真正的训练去偏方法对照

Coat 6960 条非随机训练评分 + 2065 条随机评分开发记录，2209 条随机评分保留记录不用于评分。原有 `coat_selection.py` 使用**同容量、相同 smoothed item-mean**；唯一区别是用训练集估计 propensity 进行反比加权及 cap=5/20/100 截断。

| 策略 | Brier↓ | NDCG@5↑ | Hit@5↑ |
|---|---:|---:|---:|
| unweighted | 0.153281 | 0.709910 | 0.954286 |
| estimated propensity cap=100 | 0.149546 | 0.710947 | 0.942857 |

**结论**：开发期 Brier 改善，但是 Hit 下降；来自作者提供的估计 propensity 不是已知真实随机 logging probability；此结果未在独立保留集确认，不能把校正等同于完全解决 Selection Bias。

## 2. Exposure Bias — KuaiRand 随机曝光反馈的真实训练对照

官方 KuaiRand 代码与说明：<https://github.com/chongminggao/KuaiRand>。KuaiRand-Pure 保留 7,583 个随机干预池物品和不完整的标准曝光历史；对方文档解释随机干预与标准推荐的日志收集机制。**只比较预测随机曝光下的点击，不做缺失 propensity 的伪造 IPS。**

在相同日期 2022-04-22 ～ 2022-05-08：
- 标准推荐训练日志：295497 条。
- 随机曝光日志通过事前固定 user/video/timestamp SHA256 划分：710778 条训练、238017 条随机开发、237264 条**保留不评价**。
- 模型都是 item × UI tab 的相同平滑 CTR 估计器。训练标签不来自任何随机开发/保留条目。
- 为消除训练量混杂，用固定种子从随机曝光**训练分区**再抽 295497 条构成等记录数对照。
- 对于观测支持集混杂，在标准训练与等量随机训练**共同出现的 item/tab** 5916 个键上再次比较同一批开发样本。

| 模型训练来源 | 训练曝光数 | 全随机开发 Brier↓ | 共同支持集 Brier↓ |
|---|---:|---:|---:|
| 标准日志 | 295497 | 0.223954 | 0.226810 |
| 等量随机曝光日志 | 295497 | 0.140102 | 0.145696 |
| 全随机训练分区 | 710778 | 0.138886 | 0.144494 |

**结论**：不依赖更多训练样本和缺失 item/tab 的全局回退，随机曝光训练仍明显更贴近随机曝光开发分布。这属于**真实随机参考数据上的预测去偏开发证据**，不是可部署策略的因果收益估计。两种收集模式有位置、交互方式、用户选择和未观测混杂；固定用户-物品可能重复跨时间，没有独立新用户结论。

旧 `exposure_synthetic.evaluate()` 另外保留已知 propensity 的 IPS/SNIPS 受控反证：模拟 oracle 的校验不升级为 KuaiRand 的因果可识别性。

## 3. Popularity Bias — 有真实效用代价的去热门重排

Coat 固定训练 item-mean + 真实随机评分开发候选，热门定义为非随机训练频次最高 20% 物品。不给模型用户测试标签。预设热门频率 log 惩罚系数为 0、0.05、0.1、0.2；相同用户、同候选、同目标评分，用真实随机开发 NDCG、头部曝光占比和 tail recall 同时观察。

| 惩罚系数 | Head 20% 曝光占比↓ | NDCG@5↑ | 99% NDCG guard |
|---|---:|---:|---|
| 0 | 0.21931 | 0.71095 | 通过 |
| 0.05 | 0.21724 | 0.71169 | 通过 |
| 0.10 | 0.21379 | 0.70575 | 通过 |
| 0.20 | 0.20276 | 0.69956 | **失败** |

**结论**：干预可以实实在在改变排名的热门暴露及推荐质量，而不止修改热门 Prompt；但没有证明偏好/需求不能合法解释热门曝光、全目录效用或长期人群因果公平。当前是**基于随机评分子候选的去热门重排开发实验**，不是已识别的因果 Popularity Debias。

## 4. Filter Bubble — 先否证伪纵向趋势

KuaiRand-Pure 视频标签与两个标准曝光时期（2022-04-08～04-21、04-22～05-08）；选择两期各至少 5 条有标签曝光的 17629 位用户。

- 未控制每期曝光数的用户主题 Shannon 熵，晚期减早期约 **−0.40044**。
- 把同一用户两期曝光数**配平**（预设上限每期 30 条，训练外固定抽样），熵变化约 **+0.05283**，符号反转。

**结论**：原来“推荐让兴趣变窄”的直接陈述不能成立，受观测数量差异的混杂尤其明显。当前仍缺相同初态、可操控反馈的纵向推荐对照，必须返回 `ABSTAIN_NO_MATCHED_INTERVENTION_TRAJECTORY`。这是成功的反证和可识别性改进，不是 Filter Bubble Debias。

## 5. Fairness Bias — 显式群体与失败干预

使用 Coat 官方 `user_features_map.txt` 中的 gender:men / gender:women 二元群体，事前限定公平问题为**随机评分上的预测误差（Brier）差距**，不冒充社会公平、推荐机会平等或用户性别规范性判断。

对照一：训练集估计 propensity 加权 item-mean 的相同分数；对照二：仅利用**非随机训练数据**计算群体残差，固定 shrinkage=100 做群体截距校准。公平评估只用随机评分开发数据，分群体观测完整，且同一物品排序不变。

| 指标 | 未做群体校准 | 群体校准 |
|---|---:|---:|
| men Brier | 0.152180 | 0.157710 |
| women Brier | 0.146256 | 0.149935 |
| 绝对 Brier 差距 | 0.005924 | **0.007775** |

**结论**：群体校准使两个群体的 Brier 均变差，误差差距还扩大。此方法**失败**，不准发表为 Fairness Debias 改善。该预先声明准则是群体预测误差 parity，不能外推成 Equal Opportunity。

## 当前五类 Debias 真正的完成状态

- Selection：真实开发方法对照可运行，存在 Brier / Hit 权衡；需独立验证。
- Exposure：真实随机参考数据预测对照可运行，等样本及共同支持集审计已完成；还未识别真实部署策略 OPE。
- Popularity：真实热门惩罚重排有排名/效用对照；因果不合理热门偏差尚未识别。
- Filter Bubble：两期真实数据支持审计，伪趋势已否证；**缺匹配多轮干预**，必须弃权。
- Fairness：真实群体定义与训练干预对照可运行，但方法失败；需更合适的公平定义和有效方法。

**总独立确认科学发现：0。** 这不是工程失败：我们没有把观察差异、代理指标或者失败的群体校准冒充真正的 Debias 发现。

## 与 RSI 的关系

这里每类任务已经有目标、数据、动作、指标、证据级别和失败条件。下一阶段应在保留现有 `scientific_loop` / `scientific_tree` / `scientific_director` 的同时，把这些具体**可信开发证据**接入既有 Research Memory 任务选择，让 Agent 以 Debias 目标提出有区别的假设，再评估其能否比固定基线更经济地找到合格去偏策略。不能把此 CPU 基准直接当成 RSI/LLM 科研成功。
