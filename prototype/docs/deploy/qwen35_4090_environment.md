# Qwen3.5-4B 单卡 4090 环境方案

## 目标

在单张 RTX 4090（24GB）上完成以下闭环：

1. 使用 `Qwen/Qwen3.5-4B` 生成结构化维保结果。
2. 使用 BF16 LoRA 对结构化任务进行 SFT。
3. 使用 `Qwen/Qwen3-Embedding-0.6B` 完成企业知识库检索。
4. 以本机服务方式提供生成与向量化能力。

当前阶段只做文本 SFT；图片理解作为后续案例多模态证据能力接入，不把它和首轮训练排障混在一起。

## AutoDL 规格

- GPU：RTX 4090 24GB，单卡。
- CPU：至少 8 vCPU。
- 内存：至少 64GB。
- 系统盘和数据盘合计：至少 200GB；建议 300GB。模型缓存、checkpoint、日志和仓库工作目录均放在数据盘，不占系统盘。
- 系统：Ubuntu 22.04。
- 基础镜像：`QwenLM/Qwen3.5/Qwen3.5-ALL:v1`。
- Python：3.12。
- PyTorch：2.8（镜像预装）。

不要租多卡，也不要先租长期实例。先按小时完成“首小时验收”后，再决定正式训练时长。

## 模型锁定

首次环境验证使用以下仓库和 revision。模型文件下载完成后应保留本地快照，部署时从本地路径加载。

```text
Qwen/Qwen3.5-4B
revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

Qwen/Qwen3.5-4B-Base
revision: 1001bb4d826a52d1f399e183466143f4da7b741b

Qwen/Qwen3-Embedding-0.6B
revision: 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
```

首轮 LoRA 从后训练模型 `Qwen/Qwen3.5-4B` 开始，而不是 Base：当前训练集是指令与输出格式适配数据，规模不足以让 Base 重新学会对话和结构化输出。

## 环境隔离

同一台机器上建立三个独立环境，不要混装。

```text
~/zhiweishi/env-sft       # Qwen3.5 + BF16 LoRA 训练
~/zhiweishi/env-serve     # vLLM 生成服务
~/zhiweishi/env-embed     # Qwen3 Embedding 服务或离线建库
```

原因：Qwen3.5 当前依赖 Transformers 主分支和 vLLM nightly；Embedding 有自己的稳定依赖。把它们装在同一个环境中，会让一次升级同时破坏训练、服务和建库。

## 版本策略

### 生成服务

Qwen3.5 当前需要以下组合：

```text
Python 3.12
Transformers：主分支，首次验证后锁 commit
vLLM：官方 nightly / 主分支兼容构建，首次验证后锁 wheel 或镜像 digest
PyTorch：由 vLLM 的 torch-backend=auto 选择，不手工混装
```

第一次安装后必须保存以下输出到部署记录：

```bash
python --version
python -c "import torch, transformers, vllm; print(torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__)"
pip freeze > requirements-serving.lock.txt
nvidia-smi
```

不要在已经可用的服务环境中直接 `pip install -U`。

### SFT 训练

```text
Python 3.12
PyTorch：使用 AutoDL 镜像自带、且能正常识别 4090 的 CUDA 版本
Transformers：主分支，锁 commit
PEFT：主分支，锁 commit
TRL：主分支，锁 commit
Accelerate / Datasets：与 TRL 解析出的兼容版本
```

首轮不安装 bitsandbytes，因为训练策略是 BF16 LoRA，不是 QLoRA。首轮也不强行安装 FlashAttention；先用 PyTorch SDPA 跑通加载、训练和 adapter，再单独加入 FlashAttention 优化吞吐。

当前用于首轮验证的上游代码 revision：

```text
transformers: 1048e9af78a6045444244412dfe216ba5810e7fb
peft:         036abd27e464819e0a19ddaa26093c84d5943488
trl:          4d10b20bf69314f231241cf8267a67c3241137f0
vllm:         b80ce9dd2f30913b3b054308a09bd2d86ec6202f
```

这些是“验证起点”，不是永远不升级的承诺。只有在同一张卡完成全链路 smoke test 后，才将实际安装版本写入 lock 文件。

### Embedding

```text
transformers >= 4.51.0
sentence-transformers >= 2.7.0
Qwen/Qwen3-Embedding-0.6B
```

查询侧使用固定任务指令，文档侧不加指令。例如：

```text
Instruct: Given a maintenance fault description, retrieve equipment manuals, SOPs, repair records, and approved cases that support safe diagnosis.
Query: <用户输入>
```

第一版维持 1024 维向量。不要在尚未建立检索基线前为了省存储随意降低维度。

## LoRA 训练配置

```text
训练方式：BF16 LoRA
最大长度：2048
batch size：1
gradient accumulation：8 或 16
gradient checkpointing：开启
混合精度：BF16
初始学习率：1e-4（仅作为首轮 smoke test 起点）
LoRA target：先通过 model.named_modules() 检查，再选择 all-linear 或实际线性层集合
```

不做全参数微调，不用 QLoRA。Qwen3.5 是新架构，不能直接照旧 Qwen 的 `q_proj/v_proj` 固定模块名单抄配置。

## 生成服务配置

单卡上先用 BF16 原始 4B 模型推理，不急于量化。限制上下文，避免 KV Cache 吃满显存：

```text
tensor parallel size：1
max model len：4096（稳定后可尝试 8192）
gpu memory utilization：0.80 左右
并发：先从 1 开始
```

不要使用模型卡中 262K 的上下文长度作为单张 4090 的默认配置。

## 首小时验收

租卡后按以下顺序执行，任一步失败先排环境，不开始正式训练：

1. `nvidia-smi` 确认 4090 可见、驱动和 CUDA 正常。
2. 下载并按 revision 加载 Qwen3.5-4B，完成一条纯文本 JSON 输出。
3. 用 `AutoProcessor` 加载模型，确认文本输入不需要图片也能稳定推理。
4. 打印 `named_modules()`，确认 LoRA 可注入的线性层。
5. 用现有 9 条 SFT 种子样本完成一次 1 个 epoch 的 BF16 LoRA smoke test。
6. 保存 adapter，重新加载 adapter，验证 JSON schema 输出。
7. 启动 vLLM 服务，调用一次 OpenAI 兼容接口。
8. 加载 Qwen3-Embedding-0.6B，建立并查询一个小型向量库。

全部通过后再补充审核后的数据集并开始正式训练。

## 多模态的第二步

Qwen3.5 的多模态能力不进入首轮 SFT 验收。第二阶段以案例为单位增加：

```text
现场图片 -> 图像说明 / OCR / 可见异常 -> 写入案例文本 -> Embedding 检索
命中案例 -> 展示原图 -> Qwen3.5 对关键图和文本共同复核
```

这样才能把“多模态”做成可追溯的案例证据能力，而不是只在模型名称上多一个能力标签。
