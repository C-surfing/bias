# Autonomous Bias Discovery — KuaiRand-27K 全量时序接入报告

日期：2026-10-23。服务器项目：`/data/cjh/rsi/project/`。

## 核心科学目标

让本地 Qwen 科研 Agent 在**未预定义 Bias 名称或隐藏机制答案**的前提下，从真实的、按时间连续的推荐记录中发现条件异常，提出可证伪的作用机制；通过独立样本、干预与可信偏好参照，确认偏差本身是否真实存在且损害推荐目标；再设计可逆 Debias 并保护推荐效用。日志相关性不是因果效应，不能据此擅自宣称“发现新的真实 Bias”。

## 完整数据集选取及状态

官方 KuaiRand-27K：27,285 用户、32,038,725 视频、322,278,385 次交互；完整标准日志分 4 份，总量约 23GB，基础视频特征约 2.6GB，带随机曝光参照的 1,186,059 条记录。官方压缩包 9,892,191,178 字节（约 9.21GiB），完整解压约 46GB。官方来源：https://github.com/chongminggao/KuaiRand；Zenodo：https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz。

本轮已由服务器验证下载地址/大小，开始获取压缩包。但因源站传输速度有限，**在本轮完成代码和测试时并未收到完整 27K 数据**。下载的未完成压缩文件与 aria2 元数据保存在项目 `data/external/kuairand-27k/`；不得将其视为可读取的完整 tar.gz，也不得冒称执行了 322M 次事件的大规模评估。

完整性验证先于科研读取：

```bash
cd /data/cjh/rsi/project/data/external/kuairand-27k
aria2c -c -x 8 -s 8 --file-allocation=none \
  -o KuaiRand-27K.tar.gz https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz
echo '3e3c799a24e2d23a4d2c757fbf9adf59  KuaiRand-27K.tar.gz' | md5sum -c
tar -xzf KuaiRand-27K.tar.gz
```

以上为**复现命令**，并非本轮已经完成的动作；对未完成文件不可执行 md5 校验成功声明。

## 已实现的时序 Discovery Engine

**代码：** `rsi_researcher/research/kuairand_temporal_discovery.py`

- 仅接受 4 个 `log_standard_*_27k_part*.csv` 完整日志、随机曝光 CSV、视频基础元数据和用户特征，明确拒绝 Pure 的不完整时间序列。
- DuckDB 在固定抽样方案下按用户和毫秒时间排序，构建前序反馈、最近 8 次与此前 32 次点击响应、连续重复视频、标签变化、超长时间间隔、近期响应下降、前一天其他用户对同视频的负反馈等可组合观测结构。
- 两两组合上述时间状态，不根据 Popularity/Selection/Position 等 Bias 名字提前搜索。
- 独立评估按不同用户 hash fold **并同时隔离日期**：开发 80% 用户且日期 ≤2022-04-30；确认 20% 用户且日期 ≥2022-05-01。
- 输出含有样本量、用户数、目标行为比例、对照组比例、观测差和描述性区间的研究概念卡。因多次重复点击可能来自同一用户，事件级区间只用于初筛；正式置信证据必须采用用户聚类或其它合适推断，不能直接以此作为显著性发现。
- 明确区分跨用户“同一视频前一天负面反馈”的**先验可观察关联**和可识别的“跨用户传播因果机制”；没有实际干预和共享状态控制，禁止升级为传播结论。
- 使用项目原有的 `ResearchAgent`、本地 Qwen3-8B/vLLM 和已有模型调用日志。Qwen 只收到开发异常概念卡及其支持数据，提出假设、竞争解释、可反证设计、预期效用损害、可逆 Debias 与尚缺的测量；结构化输出**冻结后**独立评估读取留出观测。
- 确认期方向相同只算 `replicated_observational_concept`，`mechanism_identified=false`、`bias_vs_utility_reference_identified=false`、`debias_effect_verified=false`。

### 需要进一步加强的地方

当前观测特征的构建仍依赖通用可测量的时间窗、事件和元数据语法；条件组合可以自动产生，但还**不是任意自由代码/开放属性发现**。真正新颖机制需与竞争性已知解释比较，具备可信的独立干预、相关性或随机反馈基准，且能通过去偏干预改善/保护推荐效用。

离线标准日志不携带足以任意恢复反事实目标的可用概率/随机行动保证。KuaiRand 的随机插入仅提供受限视频池中的参照，不能把其随机反馈值直接套用到三千万视频的整个推荐分布。

## 当前运行与复现

程序编译和 **2 项针对性单元测试均通过**：
`PYTHONPATH=. /data/cjh/rsi/discovery_private_20261024/env/bin/python -B tests/test_kuairand_temporal_discovery.py`

测试使用一次性构造的仿真格式事件样本，只验证协议、完整数据约束、时序窗口和开发/确认边界；绝**非**真实 27K 实验结果。已有更早期 KuaiRand-Pure 探索结果在 `runs/kuairand-real-discovery-20261023-v1/`，不能替代全量时序发现。

完整数据准备就绪后：

```bash
cd /data/cjh/rsi/project
PYTHONPATH=. /data/cjh/rsi/discovery_private_20261024/env/bin/python -B \
  -m rsi_researcher.research.kuairand_temporal_discovery \
  --dataset data/external/kuairand-27k/KuaiRand-27K \
  --output runs/kuairand27k-open-discovery-v1 \
  --modulo 32 --seed 20261023 --min-events 180
```

仅在授权的本地 Qwen 服务运行、科研协议固定后，执行：

```bash
PYTHONPATH=. /data/cjh/rsi/discovery_private_20261024/env/bin/python -B \
  -m rsi_researcher.research.kuairand_temporal_discovery \
  --dataset data/external/kuairand-27k/KuaiRand-27K \
  --output runs/kuairand27k-open-discovery-v1 \
  --review-existing --qwen-config configs/discovery_qwen_blind_20261024.json
```

上述 CLI 是后续完整数据复现契约，不是宣称本轮已执行完整 27K 的 Qwen 实验。

## 晋级有价值新 Bias 的不可跳过关卡

1. **观察**：发现此前未指定的异常切片，保留样本与查询定义，向 Agent 公开；
2. **机制假说**：Qwen 给出明确变量组合、作用方向/时间边界、竞争解释、可反驳预测；
3. **独立确认**：重新计算在用户与时段隔离的留出数据上的作用范围与不确定性；多候选探索要控制选择偏差；
4. **因果区分**：固定适当基准并执行可行的实验干预（或提供有效的识别设计）排除简单内容质量、时段、候选池和偏好混杂；
5. **有害性**：证实异常影响了独立可信推荐目标（点击/长观看或已验证偏好参照），而不是合理的个性化；
6. **可去偏性**：交给 Debias RSI 执行可逆干预；在冻结留出集同时记录偏差变化、总体效用、最差子群效用以及模型/实验成本；
7. **新颖性**：说明相对已知 Bias 文献增加了新的触发条件、交互路径、时间传播机制或原有理论不能预测的反证。

没有完整时序实验数据和真实干预结果，本轮**没有符合上述 7 个关卡的新 Bias 发现**，不能将代码成功、机制名称或显著相关性冒充完整科研成果。
