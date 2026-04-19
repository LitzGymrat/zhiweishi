# 原型启动说明

## 目标

该目录用于快速启动一个本地可演示的原型版本，优先跑通产品交互与演示闭环，再逐步替换为真实RAG链路。

## 目录概览

1. `app.py`：Streamlit演示界面
2. `src/`：检索链路、配置和数据处理代码
3. `data/`：原始样本、索引和案例输出目录
4. `docs/usage/`：组员试用和临时公网隧道说明
5. `docs/deploy/`：公共访问、服务器部署和交接说明
6. `dist/`：部署交付包输出目录，由打包脚本自动生成

## 常用说明

1. [说明文档索引](./docs/README.md)
2. [组员试用说明](./docs/usage/组员试用说明.md)
3. [临时公网隧道试用说明](./docs/usage/临时公网隧道试用说明.md)
4. [公共访问部署说明](./docs/deploy/公共访问部署说明.md)
5. [公共服务器5分钟部署](./docs/deploy/公共服务器5分钟部署.md)
6. [部署组员交接说明](./docs/deploy/部署组员交接说明.md)
7. [PoC材料包说明](./docs/deploy/PoC执行打包说明.md)

## 建议启动顺序

1. 先安装依赖
2. 配置 `DEEPSEEK_API_KEY`，可选配置 `DASHSCOPE_API_KEY`
3. 先跑通演示界面
4. 导入设备文档并构建索引

## 推荐命令

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 推荐使用 uv

当前项目更建议使用 `uv` 管理环境，原因是需要固定到 Python 3.11，以提升 `chromadb` 相关依赖在 Windows 环境下的兼容性。

```bash
uv sync
uv run streamlit run app.py
```

如果需要显式指定 Python 版本：

```bash
uv sync --python 3.11
uv run --python 3.11 streamlit run app.py
```

## 环境变量

```bash
DEEPSEEK_API_KEY=你的DeepSeek API Key
DASHSCOPE_API_KEY=你的DashScope API Key
DEEPSEEK_MODEL=deepseek-chat
ZHIWEISHI_EMBEDDING_PROVIDER=dashscope
ZHIWEISHI_EMBEDDING_MODEL=text-embedding-v3
```

## 下一步替换方向

1. 进一步优化切分策略与设备标签提取
2. 将案例回写内容重新写入知识库并支持再检索
3. 继续补充真实设备说明书、SOP和历史维修记录
