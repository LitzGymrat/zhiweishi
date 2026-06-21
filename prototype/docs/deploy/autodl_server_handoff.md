# AutoDL 服务器接手说明

## 目标与边界

这台 AutoDL 4090 的首要目标是验证本地模型路线：

```text
Qwen3.5-4B -> 文本/图像推理与本地服务
Qwen3.5-4B + BF16 LoRA -> SFT 训练链路
Qwen3-Embedding-0.6B -> 本地知识库向量化
```

它不是“直接把当前 Streamlit 原型改成本地模型版”的一键部署。当前原型仍调用 DeepSeek 和 DashScope；本地模型接入应用是后续代码工作。

本文件说明服务器接手顺序；模型版本、LoRA 配置和验收标准详见 `qwen35_4090_environment.md`。

不要使用 `prepare_server_handoff.ps1` 生成的公网部署包来进行本地模型训练：该包面向 Streamlit 公网部署，当前不包含 `data/sft/` 和 `scripts/`。AutoDL 应克隆完整 Git 仓库。

## 已选服务器规格

```text
GPU: RTX 4090 x1, 24GB VRAM
CPU: 16 cores
RAM: 120GB
Image: QwenLM/Qwen3.5/Qwen3.5-ALL:v1
OS image: Ubuntu 22.04, Python 3.12, PyTorch 2.8, CUDA 12.8
Data disk: use the 300GB paid disk; 50GB free disk alone is insufficient
```

模型专用镜像可用于首轮 Qwen3.5 加载验证，但不要在其全局 Python 环境中直接安装或升级所有依赖。训练、服务和 embedding 必须各自使用独立虚拟环境。

## 0. 登录前确认

1. SSH 已使用本机私钥连通；账户通常为 `root`，端口以 AutoDL 控制台为准。
2. 目标代码已有明确 Git commit 并已推送。服务器 `git clone` 不会得到本机未提交文件。
3. 不上传本机 `.env`、`.venv`、`data/chroma_db/`、`data/case_output/`、模型缓存或私钥。
4. 不在 Git 历史、终端共享记录或截图中暴露 API Key。

## 1. 首次登录检查

```bash
nvidia-smi
df -h
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
git --version
```

验收条件：4090 可见、`torch.cuda.is_available()` 为 `True`、300GB 数据盘有充足空间。模型、HF 缓存、训练 checkpoint、代码工作目录和日志都放在数据盘；系统盘只保留镜像与系统文件。

## 2. 推荐目录

```bash
# AutoDL 通常将数据盘挂载在 /root/autodl-tmp；先用 df -h 和 findmnt 确认。
findmnt -T /root/autodl-tmp
export DATA_ROOT=/root/autodl-tmp/zhiweishi
mkdir -p "$DATA_ROOT"/{repo,cache/huggingface,checkpoints,logs}
export HF_HOME="$DATA_ROOT/cache/huggingface"
command -v uv >/dev/null || python -m pip install uv
uv venv --python 3.12 "$DATA_ROOT/env-sft"
uv venv --python 3.12 "$DATA_ROOT/env-serve"
uv venv --python 3.12 "$DATA_ROOT/env-embed"
```

若 `findmnt` 显示数据盘不是 `/root/autodl-tmp`，将 `DATA_ROOT` 改为实际挂载点下的 `zhiweishi` 目录。可将 `HF_HOME` 与 `DATA_ROOT` 写入 `/root/.bashrc`。不要将 API Key 写入 shell 历史或仓库文件。

## 3. 获取代码

只有在目标改动已推送后才执行：

```bash
cd "$DATA_ROOT"
git clone <仓库地址> repo
cd "$DATA_ROOT/repo"
git switch codex/video-demo
git status --short
git log -1 --oneline
```

`git status --short` 应为空。若服务器需要某项尚未推送的改动，先在本机做范围明确的 commit 和 push；不要在服务器上手工补抄本机改动。

## 4. 现有 Streamlit 原型

现有 `prototype/pyproject.toml` 要求 Python 3.11，而 AutoDL Qwen 镜像自带 Python 3.12。因此不能直接在镜像全局环境执行 `uv sync`。

### 路径 A：只做本地模型验证

首日推荐此路径。AutoDL 只跑模型、训练和 embedding 验证；现有 Streamlit 演示继续留在本机或走项目已有 Docker 公网部署。这样可以把模型环境问题和 Web 应用问题拆开。

### 路径 B：在 AutoDL 启动现有原型

使用项目 Dockerfile，它固定为 Python 3.11：

```bash
cd "$DATA_ROOT/repo/prototype"
cp server.env.example .env
# 仅在服务器本地编辑 .env，填入必要的 API Key 和访问口令
docker compose -f compose.public.yml up -d --build
```

这条路径启动的仍是外部 DeepSeek + DashScope 版本。公网暴露前必须设置 `access_password`，并保持 `PUBLIC_APP_MODE=poc1_readonly`，除非确有需要开放完整写入演示。

## 5. Qwen3.5 首小时验收

先不要训练。按顺序执行，任一步失败都记录版本和错误后处理，不开启长任务：

1. 下载并按固定 revision 加载 `Qwen/Qwen3.5-4B`。
2. 完成一条纯文本请求，要求严格 JSON 输出。
3. 用 `AutoProcessor` 验证文本输入可稳定运行。
4. 输出 `named_modules()`，确认真实 LoRA 注入层。
5. 用现有 9 条种子样本完成 1 epoch 的 BF16 LoRA smoke test。
6. 保存 adapter，重新加载 adapter，再验证 JSON 输出。
7. 启动 vLLM，调用 OpenAI 兼容接口。
8. 加载 `Qwen/Qwen3-Embedding-0.6B`，完成小型索引建库与查询。

服务环境由 vLLM 官方方式自行解析匹配的 torch：

```bash
source "$DATA_ROOT/env-serve/bin/activate"
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

不要把该命令安装到镜像全局 Python，也不要在已正常工作的环境中无目标地执行 `pip install -U`。

## 6. SFT 数据与训练

本次服务器环境初始化提交不包含 `data/sft/` 训练数据。数据会在后续合成和审核完成后单独推送或上传；在此之前，服务器只做模型与环境验收。

数据到位后，先在独立 SFT 环境中校验：

```bash
cd "$DATA_ROOT/repo/prototype"
source "$DATA_ROOT/env-sft/bin/activate"
uv pip install openai pydantic-settings
python scripts/validate_sft_dataset.py
python scripts/export_sft_messages.py
```

如果环境缺少应用依赖，应在专门的 SFT 环境中安装，而不是改动主镜像。种子样本仅用于 pipeline smoke test；没有 SME 审核和新增回写案例对照数据前，不开始正式训练和效果宣称。

## 7. 需要保留的产物

每次环境验收至少保存：

```text
$DATA_ROOT/logs/nvidia-smi.txt
$DATA_ROOT/logs/versions.txt
$DATA_ROOT/logs/smoke-test.log
$DATA_ROOT/checkpoints/<run-id>/
```

`versions.txt` 至少包含：

```bash
python --version
python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)"
pip freeze
```

在首轮成功前，不删除缓存、不覆盖旧 checkpoint，也不运行大规模训练。

## 8. 常见误区

1. 模型能下载，不等于本地应用已经接入模型；当前 Streamlit 仍是外部 API 链路。
2. Python 3.12 模型镜像，不等于可直接运行当前 Python 3.11 原型依赖。
3. Qwen3.5 的 262K 上下文不是单张 4090 的默认设置；服务先限制到 4096。
4. 案例图片第二阶段先做 OCR、图像说明和可见异常的文本化检索；命中后再让 Qwen3.5 对关键原图复核。
5. 不要执行 `git add .`；项目目录中存在资料、导出物和环境文件，应只提交明确需要的文件。

## 9. 当日完成标准

1. 能稳定 SSH 登录并保存主机信息。
2. 代码来源、分支和 commit 可追溯。
3. Qwen3.5-4B 在 4090 上能完成一次本地文本 JSON 输出。
4. vLLM 或 Transformers 至少一条本地推理链路跑通。
5. Qwen3 Embedding 能完成一条建库和查询。
6. 已记录版本、显存占用、命令和失败信息。

达到这些标准后，再添加正式 LoRA 训练脚本、补充审核数据，并将本地服务接入 Streamlit。
