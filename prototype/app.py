from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import AppConfig, get_env_help_text
from src.corpus_manager import (
    CaseWritebackValidationError,
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    DOCUMENTS_DIR,
    SUPPORTED_SUFFIXES,
    build_corpus_summary,
    collect_corpus_documents,
    prepare_case_writeback_payload,
    reset_demo_writebacks,
    resolve_document_dir,
    save_case_writeback,
)
from src.image_evidence import (
    ImageEvidenceService,
    ImageEvidenceValidationError,
    prepare_image_upload,
    resolve_stored_image,
)
from src.rag_pipeline import RagPipeline


@st.cache_resource(show_spinner=False)
def get_pipeline(cache_version: str = "task-prompts-v1") -> RagPipeline:
    return RagPipeline(AppConfig.from_env())


def ingest_default_documents(pipeline: RagPipeline) -> None:
    if pipeline.metadata.index_exists() or not DOCUMENTS_DIR.exists():
        return

    doc_paths = collect_corpus_documents(DOCUMENTS_DIR)
    if doc_paths:
        pipeline.ingest_documents(doc_paths)


def render_fault_result(result: dict) -> None:
    st.subheader("📋 诊断摘要", divider="blue")
    st.info(result.get("summary", "当前暂无结构化摘要。"))

    writeback_case_references = result.get("writeback_case_references", [])
    if writeback_case_references:
        with st.container(border=True):
            st.markdown("#### 📝 已命中回写案例")
            for reference in writeback_case_references:
                case_id_col, operator_col, written_at_col = st.columns(3)
                case_id_col.metric("案例编号", reference.get("case_id", "未记录"))
                operator_col.metric("处理人", reference.get("operator", "未填写"))
                written_at_col.metric("回写时间", reference.get("written_at", "未记录"))
                st.caption(f"来源：{reference.get('source_label', '回写案例')} | 人工确认后归档")

            case_note = str(result.get("matched_writeback_case_note", "")).strip()
            if case_note:
                st.success(case_note)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### 🔍 证据观察")
            for item in result.get("evidence_observations", []):
                st.markdown(f"- {item}")

        with st.container(border=True):
            st.markdown("#### ⚠️ 可能原因")
            for item in result.get("possible_causes", []):
                st.markdown(f"- {item}")

        with st.container(border=True):
            st.markdown("#### 🔧 建议排查步骤")
            for index, item in enumerate(result.get("troubleshooting_steps", []), start=1):
                st.markdown(f"**{index}.** {item}")

    with col2:
        with st.container(border=True):
            st.markdown("#### 🚨 风险提示")
            for item in result.get("risk_notes", []):
                st.markdown(f"- {item}")

        with st.container(border=True):
            st.markdown("#### 💡 处理建议")
            st.write(result.get("escalation_advice", "当前暂无处理建议。"))

        with st.container(border=True):
            st.markdown("#### 🎓 培训提示")
            for item in result.get("training_points", []):
                st.markdown(f"- {item}")

    if result.get("uncertainty_note"):
        st.warning(f"**不确定性说明：** {result['uncertainty_note']}")

    with st.expander("📚 依据文档", expanded=False):
        for item in result.get("evidence_sources", []):
            st.markdown(f"- `{item}`")

    image_references = result.get("image_evidence_references", [])
    if image_references:
        with st.expander("🖼️ 检索命中图片证据", expanded=False):
            st.caption("图片派生内容只描述可见事实，仍需结合原图和现场人工复核。")
            image_columns = st.columns(min(len(image_references), 2))
            for index, reference in enumerate(image_references):
                with image_columns[index % len(image_columns)]:
                    image_path = resolve_stored_image(str(reference.get("storage_ref", "")))
                    if image_path:
                        st.image(str(image_path), caption=f"{reference.get('image_id', '图片证据')} | {reference.get('source_label', '')}")
                    else:
                        st.warning(f"{reference.get('image_id', '图片证据')} 的原图文件不可用。")


def render_training_result(result: dict) -> None:
    st.subheader("🎯 培训目标", divider="green")
    st.success(result.get("training_goal", "当前暂无培训目标。"))

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### 🗝️ 关键要点")
            for item in result.get("key_points", []):
                st.markdown(f"- {item}")

        with st.container(border=True):
            st.markdown("#### ❌ 易错点预警")
            for item in result.get("common_mistakes", []):
                st.markdown(f"- {item}")

    with col2:
        with st.container(border=True):
            st.markdown("#### 📝 训练题")
            for index, item in enumerate(result.get("quiz_questions", []), start=1):
                st.markdown(f"**{index}.** {item}")

        with st.container(border=True):
            st.markdown("#### ✅ 上岗检查清单")
            for item in result.get("assessment_checklist", []):
                st.markdown(f"- [ ] {item}")

    if result.get("coach_tip"):
        st.info(f"**👨‍🏫 带教提示**：{result['coach_tip']}")

    if result.get("uncertainty_note"):
        st.warning(f"**提示**：{result['uncertainty_note']}")

    with st.expander("📚 依据文档", expanded=False):
        for item in result.get("evidence_sources", []):
            st.markdown(f"- `{item}`")


def render_case_review(result: dict) -> None:
    st.subheader("🏷️ 归档标题", divider="blue")
    st.info(result.get("case_title", "当前暂无归档标题。"))

    with st.container(border=True):
        st.markdown("#### 📌 根因概括")
        st.write(result.get("root_cause_summary", "当前暂无根因概括。"))

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### 🔄 可复用经验")
            for item in result.get("reusable_lessons", []):
                st.markdown(f"- {item}")

        with st.container(border=True):
            st.markdown("#### 🔖 归档标签")
            tags = result.get("archive_tags", [])
            if tags:
                st.markdown(" ".join([f"`{tag}`" for tag in tags]))
            else:
                st.write("暂无标签")

    with col2:
        with st.container(border=True):
            st.markdown("#### 📖 SOP更新建议")
            for item in result.get("sop_update_suggestions", []):
                st.markdown(f"- {item}")

        with st.container(border=True):
            st.markdown("#### ⚠️ 归档缺口提示")
            for item in result.get("missing_information", []):
                st.markdown(f"- {item}")

    if result.get("review_note"):
        st.warning(f"**审核提示**：{result['review_note']}")

    with st.expander("📚 依据文档", expanded=False):
        for item in result.get("evidence_sources", []):
            st.markdown(f"- `{item}`")


def require_access_gate(config: AppConfig) -> None:
    if not config.has_access_password:
        return

    if st.session_state.get("access_granted"):
        return

    st.warning("当前演示已设置访问口令。输入正确口令后才能进入页面。")
    with st.form("access_gate_form", border=True):
        entered_password = st.text_input("🔐 访问口令", type="password")
        submitted = st.form_submit_button("进入演示", use_container_width=True)

    if submitted:
        if entered_password == config.access_password:
            st.session_state["access_granted"] = True
            st.rerun()
        else:
            st.error("访问口令不正确。")

    st.stop()


st.set_page_config(page_title="智维师 Demo", page_icon="🛠️", layout="wide", initial_sidebar_state="expanded")
st.title("🛠️ 智维师 - 维保辅助原型")
st.caption("面向制造业老旧设备维保场景的知识辅助演示原型")

pipeline = get_pipeline("ui-modes-v1")
config = pipeline.config
require_access_gate(config)
if not config.is_poc1_readonly:
    ingest_default_documents(pipeline)

index_ready = pipeline.metadata.index_exists()
corpus_summary = build_corpus_summary()
upload_category_options = [key for key in CATEGORY_ORDER if key != "uploaded"]

st.sidebar.title("项目导航")
if config.is_poc1_readonly:
    st.sidebar.success("当前模式：PoC1 只读体验")
    st.sidebar.info("只保留 PoC1 相关的故障排查体验页面，已隐藏案例回写和数据导入，也不会自动建库写盘。")
else:
    st.sidebar.info("聚焦老旧设备维保场景，展示排障、培训和案例沉淀的基本闭环。")
    if config.demo_reset_enabled:
        with st.sidebar.expander("🎬 演示拍摄控制", expanded=True):
            st.caption("拍摄前点击一次，可恢复到“回写前”的干净状态。")
            if st.session_state.get("demo_reset_message"):
                st.success(st.session_state.pop("demo_reset_message"))
            if st.button("重置回写数据并重建知识库", type="primary", use_container_width=True):
                with st.spinner("正在清理演示回写并重建知识库..."):
                    reset_result = reset_demo_writebacks()
                    ingest_result = pipeline.rebuild_default_corpus(DOCUMENTS_DIR)
                for key in (
                    "last_fault_result",
                    "last_training_result",
                    "last_training_signature",
                    "last_case_review",
                ):
                    st.session_state.pop(key, None)
                st.session_state["demo_reset_message"] = (
                    "已恢复演示状态："
                    f"删除 {reset_result['deleted_knowledge_docs']} 份回写案例、"
                    f"{reset_result['deleted_audit_files']} 份审计记录、"
                    f"{reset_result['deleted_images']} 张归档图片；"
                    f"当前知识库 {ingest_result.document_count} 份文档、{ingest_result.chunk_count} 个文本块。"
                )
                st.rerun()

with st.sidebar.expander("⚙️ 系统状态与配置", expanded=False):
    st.metric("知识库文档数", corpus_summary["total_documents"])
    st.metric("覆盖设备数", corpus_summary["device_count"])
    st.metric("索引文本块数", pipeline.metadata.chunk_count())
    st.divider()
    st.caption(f"**推理模型**：{config.deepseek_model}")
    st.caption(f"**向量模型**：{config.embedding_model}")
    st.caption(f"**存储路径**：{config.chroma_dir}")
    st.caption(f"**默认Top-K**：{config.top_k}")

with st.sidebar.expander("🔑 环境变量说明", expanded=False):
    st.code(get_env_help_text(), language="text")

tab_specs = [
    ("fault", "🔧 故障排查卡"),
    ("overview", "📊 项目概览"),
    ("dataset", "📂 测试集"),
]
if not config.is_poc1_readonly:
    tab_specs = [
        ("fault", "🔧 故障排查卡"),
        ("writeback", "📝 案例回写"),
        ("training", "🎓 培训模式"),
        ("overview", "📊 项目概览"),
        ("dataset", "📂 测试集"),
        ("ingest", "⚙️ 数据导入"),
    ]

tabs = st.tabs([label for _, label in tab_specs])
tab_map = {key: tab for (key, _), tab in zip(tab_specs, tabs)}

with tab_map["overview"]:
    st.subheader("💡 当前版本可以展示什么", divider="blue")
    if config.is_poc1_readonly:
        st.info("当前运行在 PoC1 只读体验模式，适合给组员试用或在局域网内共享故障排查流程。")
    st.markdown(
        """
1. **文档导入与整理**：支持将说明书、SOP、日志等资料纳入知识库。
2. **检索与引用**：结合关键词和向量检索，返回与当前问题相关的资料片段。
3. **结构化排障结果**：围绕故障现象生成可能原因、处理步骤、风险提示和依据文档。
4. **培训辅助**：根据排障结果整理岗位要点、易错点和训练题。
5. **案例回写**：将一次处理过程沉淀为案例，并更新到知识库中。
        """
    )
    st.subheader("⚙️ 当前实现方式", divider="blue")
    st.markdown(
        """
- **检索后生成**：先找相关资料，再组织结构化结果，尽量减少脱离资料的回答。
- **分任务输出**：排障、培训和案例复盘分别使用不同的输出模板。
- **保留依据**：回答会同时展示参考文档，便于人工复核。
- **保留兜底逻辑**：模型不可用时，系统仍可基于检索结果给出基础建议。
        """
    )

with tab_map["dataset"]:
    st.subheader("📂 维保知识库（当前状态）", divider="green")
    st.info("当前知识库主要包含设备说明书、SOP、点检日志、维保日志、维修记录和案例卡。")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("📚 知识库文档数", str(corpus_summary["total_documents"]))
    metric_col2.metric("🏭 覆盖设备种类", str(corpus_summary["device_count"]))
    metric_col3.metric("📑 文档资源类别", str(corpus_summary["category_count"]))
    metric_col4.metric(
        "📝 案例卡",
        str(next((item["count"] for item in corpus_summary["categories"] if item["key"] == "case_cards"), 0)),
    )

    st.write(f"📁 **当前文档根目录**：`{DOCUMENTS_DIR}`")

    for category in corpus_summary["categories"]:
        with st.expander(f"📌 {category['label']} ({category['count']} 份)", expanded=False):
            for record in category["records"]:
                st.markdown(f"- **{record['device_name']}** | `{record['file_name']}`")

if "ingest" in tab_map:
    with tab_map["ingest"]:
        st.subheader("📥 文档导入与索引管理", divider="orange")
        st.write("支持导入说明书、SOP、日志等资料，补充当前维保知识库。")

        with st.container(border=True):
            upload_col1, upload_col2 = st.columns(2)
            selected_upload_category = upload_col1.selectbox(
                "🗂️ 归档类别",
                options=upload_category_options,
                format_func=lambda item: CATEGORY_LABELS.get(item, item),
            )
            selected_device_name = upload_col2.text_input("⚙️ 设备归档目录", value="液压泵站")
            uploaded_files = st.file_uploader(
                "⬆️ 选择本地文档",
                accept_multiple_files=True,
                type=[suffix.lstrip(".") for suffix in sorted(SUPPORTED_SUFFIXES)],
            )

            if st.button("🚀 导入并重建索引", type="primary", use_container_width=True):
                if uploaded_files:
                    saved_paths: list[Path] = []
                    upload_dir = resolve_document_dir(selected_upload_category, selected_device_name)
                    for uploaded in uploaded_files:
                        file_path = upload_dir / uploaded.name
                        file_path.write_bytes(uploaded.getbuffer())
                        saved_paths.append(file_path)
                    result = pipeline.rebuild_default_corpus(DOCUMENTS_DIR)
                    st.success(
                        f"✅ 已归档 {len(saved_paths)} 份新文档到 `{upload_dir}`，并完成知识库重建。当前共 {result.document_count} 份文档，{result.chunk_count} 个文本块。"
                    )
                else:
                    doc_paths = collect_corpus_documents(DOCUMENTS_DIR)
                    if not doc_paths:
                        st.warning("⚠️ 当前没有可导入的维保文档，请先在下方上传文档。")
                    else:
                        result = pipeline.ingest_documents(doc_paths)
                        st.success(
                            f"✅ 已完成导入。处理了 {result.document_count} 份文档，生成 {result.chunk_count} 个文本块。"
                        )

        st.markdown("#### 📉 当前索引状态")
        colA, colB = st.columns(2)
        colA.metric("已解析文档", pipeline.metadata.document_count())
        colB.metric("文本块数量", pipeline.metadata.chunk_count())

with tab_map["fault"]:
    st.subheader("🔧 故障排查", divider="blue")
    if not index_ready:
        st.warning("⚠️ 当前未检测到已构建知识库。若只是体验 PoC1，请先在原型目录运行 `uv run python build_knowledge_base.py`，再重新打开界面。")

    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        device_name = col1.text_input("🏭 设备名称", value="液压泵站")
        symptom_name = col2.text_input("⚠️ 故障现象", value="压力波动且伴随异响，设备面板报警。")
        default_top_k = min(max(config.top_k, 3), 12)
        top_k = st.slider("🔍 检索片段数量", min_value=3, max_value=12, value=default_top_k, help="控制本次参考的资料片段数量")

    if st.button("⚡ 生成排障结果", type="primary", disabled=not index_ready):
        with st.spinner("正在检索相关资料并生成排障建议..."):
            result = pipeline.answer_fault_question(device_name=device_name, symptom=symptom_name, top_k=top_k)
        st.session_state["last_fault_result"] = result
        st.session_state.pop("last_training_result", None)
        st.session_state.pop("last_training_signature", None)
        render_fault_result(result)

        with st.expander("🕵️‍♂️ 检索命中片段", expanded=False):
            for index, hit in enumerate(result["retrieved_chunks"], start=1):
                st.markdown(f"**命中片段 {index}**")
                st.caption(f"**来源**：`{hit['source_label']}` | **分类**：{hit.get('doc_category', '未知')} | **检索得分**：{hit['hybrid_score']:.4f}")
                st.write(hit["content"])
                st.divider()
    elif "last_fault_result" in st.session_state:
        render_fault_result(st.session_state["last_fault_result"])

if "training" in tab_map:
    with tab_map["training"]:
        st.subheader("🎓 培训与带教", divider="green")
        st.info("根据排障结果整理学习要点、易错点和训练题，便于新人培训和班组带教。")
        last_result = st.session_state.get("last_fault_result")
        if not last_result:
            st.warning("⚠️ 请先在【故障排查卡】中完成一次现场故障诊断。")
        else:
            training_signature = json.dumps(
                {
                    "device": last_result.get("device", ""),
                    "symptom": last_result.get("symptom", ""),
                    "sources": last_result.get("evidence_sources", []),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if st.session_state.get("last_training_signature") != training_signature:
                with st.spinner("正在整理培训要点..."):
                    st.session_state["last_training_result"] = pipeline.build_training_support(last_result)
                st.session_state["last_training_signature"] = training_signature

            render_training_result(st.session_state.get("last_training_result", {}))

if "writeback" in tab_map:
    with tab_map["writeback"]:
        st.subheader("📝 案例回写", divider="red")
        st.write("用于演示排障结束后，如何在人工补全处理记录后，把案例整理并写回知识库。")
        st.caption("归档前需要补全实际处理步骤、最终结论和经验总结；系统会自动过滤前台生成案例，避免案例互相引用。")
        default_result = st.session_state.get("last_fault_result", {})

        with st.form("case_writeback_form", border=True):
            col1, col2 = st.columns(2)
            device_name = col1.text_input("🏭 维修设备", value=default_result.get("device", "液压泵站"))
            symptom_name = col2.text_input("⚠️ 触发症状", value=default_result.get("symptom", "压力波动且伴随异响"))
            actual_steps = st.text_area("🔧 实际确认与处理步骤", height=100)
            final_result = st.text_area("🏁 最终处理结论", height=80)
            experience_summary = st.text_area("💡 经验总结", height=80)
            operator_name = st.text_input("👨‍🔧 责任操作人", value="李工")
            case_images = st.file_uploader(
                "🖼️ 现场图片证据（可选，JPEG/PNG）",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="原图会与案例一起归档；若配置视觉模型，将抽取 OCR、图像说明和可见事实。",
            )
            if config.has_vision_endpoint:
                st.caption(f"图片将发送至已配置的视觉模型 `{config.vision_model}` 生成派生证据；结果仍须人工复核。")
            else:
                st.caption("当前未配置视觉模型：图片会安全归档，但不会自动生成 OCR 或图像说明。")
            submit = st.form_submit_button("📤 保存案例并更新知识库", use_container_width=True)

        if submit:
            payload = {
                "device": device_name,
                "symptom": symptom_name,
                "actual_steps": actual_steps,
                "final_result": final_result,
                "experience_summary": experience_summary,
                "operator": operator_name,
                "possible_causes": default_result.get("possible_causes", []),
                "recommended_steps": default_result.get("troubleshooting_steps", []),
                "evidence_sources": default_result.get("evidence_sources", []),
            }
            try:
                prepared_payload = prepare_case_writeback_payload(payload)
                prepared_images = [
                    prepare_image_upload(
                        uploaded.name,
                        uploaded.getvalue(),
                        max_bytes=config.image_max_upload_bytes,
                        max_pixels=config.image_max_pixels,
                    )
                    for uploaded in case_images or []
                ]
            except CaseWritebackValidationError as error:
                error_lines = "\n".join([f"- {item}" for item in error.errors])
                st.error(f"请先补完整案例内容后再入库：\n{error_lines}")
            except ImageEvidenceValidationError as error:
                st.error(f"图片证据未通过归档校验：{error}")
            else:
                with st.spinner("正在保存案例并更新知识库..."):
                    saved_paths = save_case_writeback(
                        prepared_payload,
                        image_uploads=prepared_images,
                        image_evidence_service=ImageEvidenceService(config),
                    )
                    pipeline.rebuild_default_corpus(DOCUMENTS_DIR)
                    case_review = pipeline.review_case_writeback(
                        saved_paths["case_payload"],
                        retrieved_chunks=default_result.get("retrieved_chunks"),
                    )
                st.session_state["last_case_review"] = case_review
                st.success(f"🎉 已保存案例，并归档到 `{saved_paths['knowledge_doc_path']}`。")
                with st.expander("🛠️ 查看保存的结构化数据", expanded=False):
                    st.json(saved_paths["case_payload"])
                    st.write(f"`{saved_paths['audit_json_path']}`")

                if saved_paths["image_evidence"]:
                    st.caption(f"已归档 {len(saved_paths['image_evidence'])} 张图片证据；派生内容已写入案例卡并参与后续检索。")

                render_case_review(case_review)
        elif st.session_state.get("last_case_review"):
            render_case_review(st.session_state["last_case_review"])
