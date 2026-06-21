# 智维师项目说明

## 项目定位

`prototype/` 是“智维师 - 知识增强智能维保辅助系统”的可运行原型。它将设备说明书、SOP、历史维修记录和人工确认的案例回写整理为可检索知识库，再生成面向现场的结构化排障卡、带教材料和案例复盘建议。

当前验证的是人机协同闭环，而不是让模型替代维保人员自动处置：

```text
资料入库 -> 混合检索 -> 基于证据生成建议 -> 人工确认与处理 -> 案例回写 -> 同类问题再次命中
```

## 当前技术链路

当前 Git 分支为 `codex/video-demo`，最近已提交版本为 `9511179 Add video demo reset workflow`。

当前应用仍使用外部服务：

```text
Streamlit UI
  -> RagPipeline
  -> Qwen/DashScope Embedding
  -> Chroma 向量检索 + BM25 关键词检索
  -> DeepSeek 结构化 JSON 生成
  -> 排障卡 / 培训卡 / 案例复盘
```

本地 `Qwen3.5-4B + BF16 LoRA + Qwen3-Embedding-0.6B` 是正在准备的下一阶段闭源部署路线。它当前尚未接入 `src/config.py` 和 `src/llm_client.py`，所以启动现有 Streamlit 服务不会自动改为调用本地 Qwen。

## 已有功能

| 功能 | 当前行为 | 主要代码 |
| --- | --- | --- |
| 故障排查 | 根据设备与故障描述检索资料，输出诊断摘要、证据、原因、步骤、风险和不确定性说明 | `app.py`、`src/rag_pipeline.py` |
| 案例回写 | 保存人工填写的处理过程为案例卡和审计 JSON，并重新入库 | `src/corpus_manager.py` |
| 回写案例引用 | 命中人工确认案例时显示案例编号、处理人和回写时间 | `src/llm_client.py`、`src/rag_pipeline.py` |
| 培训模式 | 将排障依据转为带教要点、易错点和检查清单 | `src/llm_client.py` |
| 案例复盘 | 对待归档处置记录生成根因概括、经验和待补信息 | `src/llm_client.py` |
| 演示重置 | 删除前台自动生成的回写案例和审计记录，恢复基础库 | `src/corpus_manager.py` |

## 目录结构

```text
prototype/
├─ app.py                         # Streamlit 页面与交互流程
├─ build_knowledge_base.py        # 全量导入 documents 并重建索引
├─ src/
│  ├─ config.py                   # 环境变量、运行模式、索引路径
│  ├─ rag_pipeline.py             # 文档切分、检索、生成链路
│  ├─ embeddings.py               # DashScope/Qwen embedding
│  ├─ hybrid_retriever.py         # BM25 稀疏检索
│  ├─ llm_client.py               # DeepSeek JSON prompt 与解析
│  ├─ corpus_manager.py           # 案例回写、演示重置
│  └─ loaders.py                  # PDF、DOCX、表格、文本加载
├─ data/
│  ├─ documents/                  # 说明书、SOP、记录、案例卡等源知识
│  ├─ chroma_db/                  # Chroma 索引，不入 Git
│  ├─ bm25_index.json             # BM25 索引
│  ├─ case_output/                # 回写审计 JSON，不入 Git
│  └─ sft/                        # 本地 4B 模型 SFT 数据与说明
├─ scripts/                       # SFT 校验、导出、教师草稿脚本
└─ docs/                          # 项目、部署和交接文档
```

## 数据与演示回写

基础知识库位于 `data/documents/`。前台回写成功后会新增：

```text
data/documents/case_cards/<设备>/case_YYYYMMDD_HHMMSS.md
data/case_output/<设备>/case_YYYYMMDD_HHMMSS.json
```

前者是可检索案例卡，包含案例编号、处理人、回写时间、实际步骤、结论和经验；后者是原始审计记录。演示重置只删除上述自动生成的 `case_*.md` 与 `case_*.json`，不会删除基础说明书、SOP 和既有案例卡。

因此可做可控对比：重置 -> 查询 -> 回写人工经验 -> 用同一问题再次查询 -> 命中并展示回写案例信息。

## 配置与运行模式

核心配置在 `.env`，模板见 `.env.example` 和 `server.env.example`。真实 API Key、模型缓存、案例审计记录均不得提交到 Git。

```text
deepseek_api_key=...              # 当前结构化生成服务
qwen_api_key=...                  # 当前 DashScope embedding 服务
qwen_embedding_name=text-embedding-v4
app_mode=full 或 poc1_readonly
access_password=...               # 对外访问时必须设置
demo_reset_enabled=true           # 仅演示场景显示重置按钮
```

`app_mode=full` 包含案例回写、培训和数据导入；`poc1_readonly` 用于公共只读展示。Docker 公网部署通过 `PUBLIC_APP_MODE` 传入运行模式，默认只读。

## SFT 数据状态

`data/sft/` 的目标是让本地 4B 模型学会“将 RAG 证据转换为稳定 JSON”，不是把设备事实硬塞进模型参数。当前这些数据仍在本地整理，**不属于本次 AutoDL 环境初始化提交**；后续应在合成、审核完成后单独提交或上传。

- 当前本地工作区有 9 条种子样本，状态均为 `evidence_checked_pending_sme`。
- 它们只能用于训练链路 smoke test、格式稳定性验证和基线实验，不能称为正式生产数据。
- 当前缺少真实“前台案例回写”命中/未命中样本；正式训练前须补充至少 20 组人工审核对照数据。
- 可用 `python scripts/validate_sft_dataset.py` 校验 schema、来源、角色顺序和审计规则。

详细规则见 `data/sft/README.md` 与 `data/sft/schema.md`。

## 关键边界

1. 当前原型依赖外部 DeepSeek 与 DashScope；本地 Qwen 尚未接入应用。
2. 当前 `pyproject.toml` 限定 Python `>=3.11,<3.12`，用于现有 Streamlit/Chroma 原型。
3. AutoDL 的 Python 3.12 Qwen 镜像不能直接用于 `uv sync` 启动该原型；原型应使用 Docker 的 Python 3.11，或单独准备 Python 3.11 环境。
4. 本地 Qwen 的训练、服务和 embedding 环境必须与 Streamlit 原型隔离。
5. 当前工作区存在未提交的开发和资料文件。部署前只能提交明确需要的文件，禁止 `git add .`。
6. `prepare_server_handoff.ps1` 生成的是公网原型部署包，当前不包含 `data/sft/` 和 `scripts/`；AutoDL 本地模型工作必须克隆完整仓库。

## 接手阅读顺序

1. 服务器准备与本地模型验收：`docs/deploy/autodl_server_handoff.md`
2. Qwen3.5 单卡版本与配置：`docs/deploy/qwen35_4090_environment.md`
3. 现有公网只读原型部署：`docs/deploy/部署组员交接说明.md`
