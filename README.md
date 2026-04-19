# 智维师项目工作区

本工作区已经按当前确认方向整理为两部分：

1. `docs/`：立项、背景、技术设计、MVP任务和演示脚本。
2. `prototype/`：技术启动脚手架，用于快速搭建本地演示原型和维保知识库闭环。

当前确认的项目方向：

- 项目名称：智维师
- 正式题目：面向东北中小制造企业老旧设备维保场景的知识增强智能辅助系统
- 核心定位：围绕老旧或非标设备维保中的知识分散、经验依赖、新人上手慢等问题，构建兼顾故障排查、培训学习和经验沉淀的本地化辅助系统。

文件说明：

- `docs/01_project_brief.md`：定稿方案与项目定位
- `docs/02_evidence_summary.md`：背景材料与当前证据边界
- `docs/03_technical_design.md`：技术路线与系统设计
- `docs/04_mvp_tasks.md`：两周内的开发任务拆解
- `docs/05_demo_script.md`：答辩演示脚本与讲述顺序
- `docs/06_real_data_pack.md`：公开真实数据、来源和建议用法
- `prototype/README.md`：技术原型启动说明
- `prototype/app.py`：可直接扩展的本地演示界面

如果你当前分到的任务只是“把原型部署到服务器”，直接看 `prototype/部署组员交接说明.md` 即可，不需要先通读 `docs/`。如果要先整理一份不带 `.env` 的干净交付包，直接运行 `prototype/prepare_server_handoff.bat`。

当前 demo 使用的知识库数据，已经从“背景报告”切回到维保场景测试集，目录位于 `prototype/data/documents/`，按说明书、SOP、点检日志、维保日志、维修记录、案例卡分类整理。

建议推进顺序：

1. 先看 `docs/03_technical_design.md`，统一技术方案边界。
2. 再按 `docs/04_mvp_tasks.md` 分工，开始原型实现。
3. 用 `prototype/app.py` 先跑通演示壳，再逐步替换为真实检索与问答链路。
4. 准备答辩材料时，用 `docs/06_real_data_pack.md` 讲背景，用 `prototype/data/documents/` 和案例回写闭环讲 demo 本体。
