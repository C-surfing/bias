# D1 输入来源

作者：[AgentCF-WWW](https://github.com/leoleojie/AgentCF-WWW)。固定版本：`71ff3b5bce69432556884c49fe7f19528b6756ec`。作者说明代码采用 Apache 2.0，项目代码和数据仅用于学术用途；原说明与许可保存在 `vendor/AgentCF-reference`。

- `CDs.item`：从该固定版本的 `agentcf/dataset/CDs-100-user-dense/CDs.item` 下载，3,866,165 字节。
- `CDs-100-user-dense.train.inter` / `valid.inter` / `random`：来自此前对同一作者版本的本机审计快照。已补齐固定版本的源码目录作参考；本次不读取原 test 标签。
- `author_prompts.json`：直接提取上述固定版本 `agentcf/props/AgentCF.yaml` 的 7 个 basic 选择、反思和排名模板；没有拷入 API 密钥配置。

本次工程检查选择 train 中前 2 用户，每人前 2 事件，按“所有人的第 1 条，再所有人的第 2 条”运行。它不是经过确认的全局真实时间线。负候选按完整训练分区频次抽样，只排除当时已观察的历史和当前正物品；随机种子为 20261013。验证使用同一作者候选池中前 9 个唯一且非目标编号，加 valid 目标，固定随机打乱。不同于作者直接运行 test，此处仅使用 valid 检查工程链路。

本次两轮、每事件串行更新，不执行原版批量并行。基本提示词、用户→正物品→其他用户的共享路径保留；输出格式统一为 JSON 和唯一 ID。它是现代接口下的同机制小实现，不是对原版论文数值的复现。源代码的工程修正不算算法收益。
