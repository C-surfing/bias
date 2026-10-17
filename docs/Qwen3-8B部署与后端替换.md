# Qwen3-8B 部署与研究工作流接入

这份流程对应实验室服务器和 Qwen 官方 `Qwen3-8B`，目标是用自部署模型承担研究计划与总结，推荐器和独立评分继续由普通程序执行。把模型换成本地服务，不等于去偏效果自然变好。

## 1. 服务器实测配置与范围

2026-10-15 已成功 SSH 登录并只读检查：Ubuntu Linux，驱动 580.159.04，Python 3.10.12，内存约 503 GiB；`/data` 可用约 5.5 TiB。共有 5 张 A100-SXM4-80GB、2 张 A100-PCIE-40GB。检查时 4 号卡正在使用，5 号 40GB 卡空闲；空闲状态必须在每次启动时重新查。

这次选择一张 40GB 卡，UUID `GPU-e9bc2d1d-cb92-bc3e-b145-855b0cddae0b`。模型权重本身约为 8.2B 参数 × 2 字节，另需要缓存上下文和运行空间；最终是否足够以服务实际启动为准。先用原精度 BF16，不增加量化这一变化因素。BF16 是每个数约占 2 字节的存储方式。

所有项目、模型、环境和主要缓存放在 `/data/cjh/rsi`。不使用 sudo、不安装系统驱动、不改系统 Python、不执行 conda init，不读写其他人的项目或环境。模型只提交规定格式的研究建议，没有 shell、SSH、文件编辑或安装工具；程序只执行已经实现的推荐动作。

## 2. 除了 vLLM，还需要什么

| 部分 | 做什么、为什么需要 | 本次实现 |
|---|---|---|
| 可用驱动 | 让程序使用 GPU；与软件自带 CUDA 运行库兼容 | 读取现有驱动，禁止升级系统驱动 |
| 独立环境 | 避免与别人或研究代码互相改坏依赖 | 自己的 Miniforge；inference 和 research 两个环境 |
| 官方权重 | 保存实际使用的模型版本 | Hugging Face 或 Qwen 官方 ModelScope；保存版本或文件校验值 |
| 模型服务 | 把 GPU 推理变成可调用接口 | vLLM，单卡，只监听服务器 127.0.0.1 |
| 后端适配 | 两种模型的请求格式不同，不能只改名字 | `runtime/models.py` 工厂和 VLLMModel |
| 输出与长度控制 | 模型必须返回可执行研究动作，长历史不能无限增长 | JSON Schema、本地检查、输出上限、请求/时间预算 |
| 连接与运行管理 | 本机访问、断开 SSH 后服务存活、定位故障 | SSH 隧道、自己的 nohup 进程、日志和启动/停止脚本 |
| 接入与研究检查 | 能聊天不代表能研究，更不代表去偏成功 | 服务检查 → 实际计划/执行/总结 → 独立评分 |
| 实验记录 | 防止混用模型、数据、设置和历史结果 | 模型配置、数据指纹、冻结源码、日志及依赖版本 |

LangGraph 负责安排有状态的工作步骤，支持持久化、恢复和人工介入，不负责模型推理，也不会自动改善推荐。当前步骤已由研究流程和日志串联，先不新增这个依赖；以后需要跨天恢复、多分支研究或人工暂停点时，再将现有步骤接进图中。这里的“图”就是步骤以及它们之间的先后关系。[官方说明](https://docs.langchain.com/oss/python/langgraph/overview)

## 3. 两个代码包怎样放置

部署工具包用于建立服务；完整研究源码包用于运行研究。不要在别人的环境中执行下面命令。SSH 密码只在登录时输入，不写入文件。

```powershell
ssh -p 22 cjh@172.31.233.201
```

在自己的目录中建立以下结构；本轮已开始在这里执行部署，重用前应先看实际状态记录，避免重复启动。

```text
/data/cjh/rsi/
├── deployment/qwen3/     本包部署脚本
├── project/              第二份完整研究源码，内有 rsi_researcher、configs、requirements.txt
├── tools/miniforge/      自己的 conda 程序
├── envs/inference/       vLLM 及模型下载工具
├── envs/research/        推荐系统研究代码需要的依赖
├── models/Qwen3-8B/      模型文件
├── cache/                下载、编译、CUDA 的缓存
├── logs/                 安装、服务与依赖版本记录
└── run/                  本服务 PID 和接入检查
```

工具包中的 `project/requirements.txt` 只是研究环境依赖；完整研究包应解压到同一个 `project/`，其中的源码与数据入口不可省略。已有文件先核对版本，旧实验目录不覆盖。

## 4. 建立独立环境，下载模型

```bash
# 只在自己的目录执行
cd /data/cjh/rsi
RSI_MODEL_SOURCE=modelscope bash deployment/qwen3/bootstrap.sh
```

脚本校验 Miniforge 安装文件，创建两个独立 Python 3.12 环境，在推理环境安装固定的 `vllm==0.31.0`，在研究环境安装项目依赖，并分别保存实际依赖清单。软件携带的 GPU 运行库安装在自己的环境里，和修改系统驱动是两回事。[vLLM 安装说明](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)

本机实测服务器访问 Hugging Face 超时，因此上面使用 Qwen 官方 ModelScope。能够直连 Hugging Face 时，可省略 `RSI_MODEL_SOURCE`。两种来源不会在已有模型目录中静默互换。Hugging Face 记录不可变版本号；ModelScope 的 master 会变化，所以额外保存实际模型文件的 SHA-256，见 `models/qwen3-8b-manifest.json`。[Qwen 官方下载说明](https://qwen.readthedocs.io/en/stable/inference/transformers.html)

下载失败后先看日志和所选来源，不用关闭证书检查、改系统代理、重新装驱动或重装整个环境。脚本使用下载工具的缓存接续下载；已有半包与完整包不能混为一谈。

## 5. 重新确认一张空闲卡，再启动

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
export RSI_GPU_UUID=GPU-e9bc2d1d-cb92-bc3e-b145-855b0cddae0b
bash /data/cjh/rsi/deployment/qwen3/start.sh
tail -n 40 /data/cjh/rsi/logs/vllm.log
curl --fail http://127.0.0.1:8000/health
```

上面的 UUID 是本轮选定的卡，不是永久预约。`serve.sh` 每次启动重新检查该卡，发现占用就停止，不结束其他进程。`CUDA_VISIBLE_DEVICES` 仅让本次服务看到这一张卡；卡内的逻辑编号可能显示为 0，不代表用了服务器物理 0 号卡。

启动参数是单卡、BF16、最长 32768 tokens、最多两条并行请求、最多预留该卡 75% 显存。token 是模型处理文字的单位。模型会用一部分显存缓存已读文字，不能把 40GB 全当作权重空间。服务用 `--generation-config vllm`，实际采样设置由每次请求明确给出，避免额外继承模型文件中的默认行为。

`start.sh` 记录自己启动的服务 PID，即进程编号。安装完成、进程存在、健康接口成功、研究能完成是四个不同状态；不能只看到一个 PID 就说部署成功。

## 6. 先验证服务，再验证研究角色

部署工具包可以独立验证，不需要连接付费模型：

```bash
/data/cjh/rsi/envs/inference/bin/python /data/cjh/rsi/deployment/qwen3/probe.py
```

完整源码包再验证一次实际研究输出格式：

```bash
cd /data/cjh/rsi/project
/data/cjh/rsi/envs/research/bin/python -m rsi_researcher check-model \
  --config configs/qwen_backend_comparison.json --output runs/qwen-backend-check
```

它先查健康状态和模型列表，再发出一条带假设、竞争解释、动作、强度和失败条件的真实请求。检查不通过就看 `calls.jsonl` 的状态与返回，不能自动退回 DeepSeek。输出目录每次用新名字。

Qwen 默认可能产生思考文本；这轮用 `chat_template_kwargs.enable_thinking=false` 关闭思考，便于先核验 JSON 输出和接口。采用官方非思考示例的采样设置，另记录 seed。关闭思考是本次接入选择，不表示它对研究能力最好；之后应作为单独对照，不同时改数据、算法再宣称模型更优。[Qwen vLLM 说明](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)

DeepSeek 用原有 `submit_result` 工具格式，Qwen/vLLM 用 `response_format.json_schema`。两边都经过相同的本地字段校验，返回 JSON 合法不等于内容正确。[vLLM 结构化输出](https://docs.vllm.ai/en/stable/features/structured_outputs/)

## 7. 本机通过 SSH 隧道使用服务器模型

在 Windows 另开一个终端并保持运行：

```powershell
ssh -N -L 8000:127.0.0.1:8000 cjh@172.31.233.201
```

SSH 隧道的意思是：把本机 8000 端口的请求通过 SSH 送到服务器的 8000 端口。模型服务无需对校园网或公网开放，也不用改防火墙。若本机 8000 已占用，改成 `-L 18000:127.0.0.1:8000`，同时把配置中的 `base_url` 改成 `http://127.0.0.1:18000/v1`。适配器直连配置地址，不修改 Clash 或系统代理。

完整源码也可以直接在服务器运行。推荐训练和评分仍在 CPU，GPU 主要服务 Qwen 研究员。

## 8. 分开运行“只换模型”和“改善研究”

```bash
cd /data/cjh/rsi/project
PY=/data/cjh/rsi/envs/research/bin/python
$PY -m rsi_researcher fetch-data --dataset coat --output data/external/coat
$PY -m rsi_researcher fetch-data --dataset kuairand-pure --output data/external/kuairand-pure

# 保留旧数据范围、候选动作和研究安排，只替换后端
$PY -m rsi_researcher study --config configs/qwen_backend_comparison.json --output runs/qwen-backend-only

# 新开发用户组、保留全体历史训练数据、更细强度与新增改法
$PY -m rsi_researcher study --config configs/qwen_research_v2.json --output runs/qwen-research-v2
```

上面两批各最多 30 次本地模型请求，包含重试，不会调用 DeepSeek。自部署也有显存、GPU 时间和电费成本，不能把旧 DeepSeek 费率套到本地模型上。第一批是接入条件下的探索性模型对照；第二批同时改了研究条件，不能把其效果归因于换成 Qwen。

如果要走 AgentCF 的原研究路径，用 `run --config configs/qwen_agentcf.json --output runs/qwen-agentcf`。它和 ItemCF 真实数据研究是两个入口，运行耗时不同，不能把一条路径的效果冒充另一条。

默认 `study` 现在使用 Qwen 新配置；历史 `configs/official_study.json` 仍保留 DeepSeek 设置，用于复核原批。不要运行错配置。

## 9. 怎样看运行状态和停止

```bash
tail -n 40 /data/cjh/rsi/logs/vllm.log
nvidia-smi -i "$RSI_GPU_UUID"
cat /data/cjh/rsi/project/runs/qwen-research-v2/run_result.json
bash /data/cjh/rsi/deployment/qwen3/stop.sh
```

停止脚本只接受记录中的进程编号，并核对进程所有者和命令路径；不使用 `killall python`、`pkill vllm` 或结束显卡上全部任务。结束服务不删除模型、环境和实验记录。模型下载、依赖安装失败时，不需要为了释放空间直接删除其他目录。

本轮实际完成到哪里、使用了哪张卡、是否收到模型输出，以[服务器部署实际记录](Qwen服务器部署实际记录_2026-10-15.json)为准。研究效果与新增改法的说明见[真实流程改进说明](真实流程改进说明_2026-10-15.md)。
