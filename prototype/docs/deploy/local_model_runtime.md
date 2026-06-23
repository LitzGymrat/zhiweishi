# 本地双模型运行链路

本地链路与在线链路互斥，通过 `.env` 中的 `runtime_provider` 和 `embedding_provider` 切换：

```text
在线：Streamlit -> DeepSeek API；Streamlit -> DashScope Qwen Embedding
本地：Streamlit -> LoRA vLLM (8001)；Streamlit -> Qwen3 Embedding vLLM (8002)
```

两套 embedding 的向量空间不能混用。应用发现 embedding 配置变化时，会自动重建知识库索引；首次切换时需要等待重建完成。

## 1. 启动本地 LoRA 生成服务

```bash
source /root/autodl-tmp/zhiweishi/env-serve/bin/activate

BASE=/root/autodl-tmp/zhiweishi/cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
ADAPTER=/root/autodl-tmp/zhiweishi/checkpoints/qwen35_lora_sft

vllm serve "$BASE" \
  --host 127.0.0.1 --port 8001 \
  --max-model-len 4096 \
  --served-model-name qwen3.5-4b \
  --gpu-memory-utilization 0.80 \
  --enable-lora --lora-modules finetuned="$ADAPTER"
```

## 2. 启动本地 Embedding 服务

在第二个终端中执行。单张 4090 已同时运行上面的 4B LoRA 服务时，embedding 服务先使用 0.15 的显存配额；若 OOM，可将生成服务降至 0.75 后再重启二者。

```bash
source /root/autodl-tmp/zhiweishi/env-serve/bin/activate

EMBED=/root/autodl-tmp/zhiweishi/cache/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3

vllm serve "$EMBED" \
  --runner pooling \
  --host 127.0.0.1 --port 8002 \
  --served-model-name qwen3-embedding-0.6b \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.15
```

确认两个端点：

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8002/v1/models
```

## 3. 切换应用

宿主机直接运行 Streamlit 时，`.env` 配置如下：

```dotenv
runtime_provider=local
local_llm_base_url=http://127.0.0.1:8001/v1
local_llm_model=finetuned
local_llm_api_key=EMPTY

embedding_provider=local
local_embedding_base_url=http://127.0.0.1:8002/v1
local_embedding_name=qwen3-embedding-0.6b
local_embedding_api_key=EMPTY
```

Docker 中的 Streamlit 容器不能通过 `127.0.0.1` 访问宿主机服务，使用：

```dotenv
local_llm_base_url=http://host.docker.internal:8001/v1
local_embedding_base_url=http://host.docker.internal:8002/v1
```

`compose.public.yml` 已写入 `host.docker.internal:host-gateway` 映射。重启应用后，侧栏应显示“本地 LoRA 模型 / finetuned”和“qwen3-embedding-0.6b”。

## 4. 切回在线链路

```dotenv
runtime_provider=online
embedding_provider=qwen
```

保留原有 DeepSeek 与 DashScope 配置即可。应用会重新构建在线 embedding 对应的索引。
