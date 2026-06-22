from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_sft_dataset import (
    TASK_SCHEMAS,
    validate_assistant_output,
    validate_messages,
    validate_runtime_semantics,
)
from src.config import AppConfig
from src.llm_client import build_runtime_system_prompt, build_runtime_user_message


V3_DIR = ROOT / "data" / "sft" / "runtime_v3"
DEFAULT_MATRIX = V3_DIR / "scenario_matrix.json"
FIELD_MAPPING_PATH = V3_DIR / "field_mapping.json"
DEFAULT_MODEL = "deepseek-v4-pro"
REVIEWED_SOURCE = "来源：前台案例回写（人工确认后归档）"
DATA_CUTOFF = datetime(2026, 6, 20)
SPLIT_OFFSETS = {"train": 0, "development": 10_000, "test": 20_000}


@dataclass(frozen=True)
class Profile:
    code: str
    device_type: str
    symptom: str
    component: str
    cause: str
    evidence: str
    action: str
    risk: str
    normal_rule: str


PROFILES: tuple[Profile, ...] = (
    Profile("HPU", "液压泵站", "主回路压力波动并伴随泵体断续异响", "吸油接头与滤芯", "油位偏低、吸油侧进气或滤芯压差异常", "油位低于控制线，吸油接头锁紧不足且滤芯压差偏高", "补油、复核接头密封并更换异常滤芯后低负荷验证", "带故障升载可能加剧泵体磨损和供油不稳", "压力波动叠加断续异响时，应先排除吸空和进气，未确认前不得高负荷试运行。"),
    Profile("PMP", "离心泵", "出口压力波动且轴承振动升高", "联轴器与入口过滤器", "联轴器对中偏差、入口过滤器堵塞或叶轮失衡", "振动值超过运行基线，入口压差上升且联轴器弹性体有偏磨", "先检查入口过滤器和联轴器对中，再做低负荷联机复测", "继续运行可能造成轴承损伤或机械密封失效", "振动异常时应先检查入口工况、基础紧固和联轴器对中，不应直接拆泵。"),
    Profile("CMP", "螺杆空气压缩机", "排气温度持续升高且油压略有下降", "油冷却器与油过滤器", "油冷却器换热下降、油过滤器堵塞或润滑油劣化", "油温接近报警值，油过滤器压差上升，冷却风道积尘明显", "清理冷却风道、核对油过滤器压差并按规定换油或换芯", "高温运行可能引起润滑失效和主机损伤", "排气温度持续接近报警值时，应先核对冷却与润滑条件并准备降负荷。"),
    Profile("GBX", "皮带输送机减速机", "减速机温升异常并出现周期性啸叫", "齿轮啮合与润滑油", "润滑油不足、齿轮磨损或安装同轴度偏差", "油位偏低，啸叫随负载周期变化，箱体振动较历史值上升", "复核油位和油质，检查紧固与同轴度，必要时停机开盖检查齿面", "齿面损伤扩大后可能导致卡滞或输送中断", "温升与周期性异响同时出现时，不得通过提高带速来验证故障。"),
    Profile("INJ", "注塑机液压系统", "合模压力不稳并伴随液压泵异响", "比例阀与液压油", "比例阀阀芯卡滞、油液污染或泵吸空", "压力波动时油温偏高，回油滤芯接近更换压差，阀响应有迟滞", "先检查油液清洁度和滤芯，再在停机条件下检查比例阀动作", "强行连续生产可能造成模具受力异常和泵组损伤", "压力不稳时应先确认油液与阀控条件，禁止带故障反复高速试模。"),
    Profile("FAN", "离心风机", "风量下降且振动值逐步升高", "叶轮积灰与轴承座", "叶轮积灰失衡、轴承润滑不足或基础松动", "风量低于设定，振动趋势连续三班上升，轴承座温度接近预警线", "检查入口阻力和叶轮积灰，复核轴承润滑与基础紧固后再试运行", "振动持续上升可能引起叶轮擦碰和轴承烧损", "风量下降叠加振动趋势上升时，应按趋势停机评估，不宜长期观察等待。"),
    Profile("CNC", "数控机床主轴", "主轴温升偏高且加工表面出现周期纹", "主轴轴承与冷却回路", "轴承预紧异常、润滑不足或冷却回路流量下降", "主轴温度高于工艺基线，冷却回路流量下降且振动频谱出现特征峰", "核对冷却流量与润滑状态，限制转速并由专业人员复测轴承状态", "继续高转速加工可能造成主轴精度劣化或轴承失效", "主轴温升异常时先限制转速，未完成冷却和润滑检查前不得继续满速加工。"),
    Profile("CWP", "循环水泵", "电机电流升高且出口流量不足", "入口滤网与叶轮流道", "入口滤网堵塞、叶轮流道结垢或阀门开度异常", "电流高于同负荷基线，入口压差增大，出口流量低于工艺要求", "核对阀门开度和入口滤网，必要时停泵检查叶轮流道", "流量不足可能影响下游换热并造成电机过载", "电流升高且流量下降时，应先排查入口阻力和水力通道，不宜盲目提高频率。"),
    Profile("VFD", "变频器", "输出电流波动并间歇触发过流报警", "功率模块与电机电缆", "参数漂移、电机电缆绝缘下降或功率模块散热异常", "同一负载下输出电流波动，柜内散热温度接近预警线", "记录报警码和负载变化，检查散热、端子与绝缘数据后再复位测试", "反复带故障复位可能扩大功率模块损伤", "过流报警反复出现时，应先保存报警和负载数据，禁止连续强制复位。"),
    Profile("SER", "伺服定位系统", "定位偏差报警间歇出现且复位后短时恢复", "编码器与传动机构", "编码器信号波动、传动间隙增大或参数整定不匹配", "报警集中在加减速阶段，复位后短时正常，位置误差趋势上升", "记录报警码和误差曲线，检查编码器接头与机械间隙，再由专业人员复核参数", "持续运行可能造成撞机或定位精度失控", "间歇定位报警不得仅靠复位消除，应保留趋势和报警码后排查。"),
    Profile("ROB", "工业机器人", "末端轨迹偏移并出现间歇位置报警", "关节减速器与标定数据", "关节回差增大、标定漂移或末端负载变化", "轨迹误差在特定姿态增大，报警码与末端负载切换时间相关", "记录姿态、负载和报警码，复核标定及关节回差后再恢复自动运行", "轨迹偏移可能造成工装碰撞", "发生位置报警时应先降速并切换安全模式，未复核标定前不得恢复全速自动运行。"),
    Profile("AGV", "AGV 搬运车", "行驶电机温度升高且靠站位置偏差增大", "驱动轮与定位传感器", "驱动阻力增大、轮组磨损或定位传感器漂移", "低负荷时基本正常，高负荷搬运后电机温度和靠站误差同步上升", "记录负载、温度和定位偏差，检查轮组阻力与传感器清洁度后复测", "定位偏差可能导致站点碰撞或取放失败", "高负荷异常时应限制任务并先验证轮组和定位条件，不得仅靠重新定位掩盖问题。"),
    Profile("CT", "冷却塔", "出水温度偏高但风机电流正常", "喷淋系统与填料", "喷淋不均、填料堵塞或循环水分配异常", "风机电流稳定但出水温度偏离设定，喷淋压力低于班组基线", "检查喷淋压力和布水均匀性，必要时安排停机检查填料状态", "换热能力下降可能影响下游工艺温控", "风机电流正常不能排除换热异常，应同时核对喷淋和填料状态。"),
    Profile("HEX", "板式换热器", "压差升高且换热效率下降", "板片流道与过滤器", "板片结垢、过滤器堵塞或流量分配异常", "进出口压差高于基线，温差扩大且流量记录下降", "核对过滤器压差和流量，按制度安排清洗或拆检板片", "压差持续升高可能造成供液不足和能耗上升", "压差异常时应先确认测点和过滤器状态，避免直接判定板片堵塞。"),
    Profile("PKG", "包装机", "封口温度波动且成品密封不良", "加热回路与温控传感器", "温控传感器漂移、加热元件接触不良或控制参数失配", "温度曲线短时波动，产品抽检出现密封强度下降", "保留温度曲线和样品记录，检查传感器接线与加热回路后再调整参数", "密封不良可能造成批量质量风险", "温度波动时不得直接通过提高设定温度掩盖问题，应先核对测量与加热回路。"),
    Profile("VIS", "视觉检测设备", "误判率升高但曝光参数显示正常", "镜头、光源与算法阈值", "镜头污染、光源衰减或算法阈值与当前产品不匹配", "曝光参数未变但误判集中在特定批次和特定位置", "保存原始图像和误判样本，检查镜头清洁度、光源均匀性及阈值版本", "误判率上升可能造成大量良品拦截或漏检", "曝光参数正常不代表视觉链路正常，应保留图像证据后再调整算法阈值。"),
    Profile("PLC", "PLC 控制柜", "间歇 I/O 报警且复位后短时恢复", "I/O 模块与端子排", "端子接触不良、模块温升或现场信号干扰", "报警在高温时段更集中，复位后短时恢复，柜内模块温度偏高", "记录报警地址和发生时段，检查端子紧固、模块温度和屏蔽接地", "盲目复位可能掩盖控制回路失效风险", "间歇 I/O 报警必须保存报警地址和时段，不能把复位成功当作故障排除。"),
    Profile("DRY", "空压站干燥机", "露点升高但压缩机运行参数正常", "吸附剂与切换阀", "吸附剂失效、切换阀动作异常或再生气不足", "露点连续偏高，压缩机压力和温度正常，切换周期有偏差", "记录露点趋势和切换周期，检查再生气与切换阀动作，再评估吸附剂状态", "高露点可能影响气动元件和工艺品质", "压缩机参数正常不能排除干燥机异常，应围绕露点和切换过程补充证据。"),
)

OPERATORS = ("张师傅", "李工", "王师傅", "陈工", "周师傅", "赵工", "刘师傅", "吴工", "孙师傅", "杨工")
DATE_PATTERN = re.compile(r"20\d{2}-\d{2}-\d{2}")
STRONG_TERMS = ("根因是", "本次确认", "已确认", "确认当前", "异常消失")
TERM_ALTERNATIVES = {
    "设备不一致": ("设备不一致", "设备类型完全不同", "设备类型与当前输入不一致"),
    "症状不一致": ("症状不一致", "症状与当前", "症状描述与当前"),
    "截断": ("截断", "不完整", "缺少完整"),
    "缺少": ("缺少", "不完整", "未提供"),
    "最终结论": ("最终结论", "结论字段", "结论缺失"),
    "实际处理步骤": ("实际处理步骤", "处理步骤字段", "步骤缺失"),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(f"{text}\n" if text else "", encoding="utf-8")


def task_for_family(family: str) -> str:
    if family.startswith("fault_"):
        return "fault_diagnosis"
    if family.startswith("training_"):
        return "training_support"
    if family.startswith("case_review_"):
        return "case_review"
    raise ValueError(f"未知情景族：{family}")


def material_sequence(sequence: int, split: str) -> int:
    return SPLIT_OFFSETS[split] + sequence


def unit_name(profile: Profile, material_id: int) -> str:
    return f"{profile.device_type} {profile.code}-{material_id + 3000:05d}"


def safe_datetime(material_id: int) -> datetime:
    # 6,000 distinct timestamps fit before the declared 2026-06-20 data cutoff.
    return datetime(2026, 4, 1, 8, 0, 0) + timedelta(minutes=17 * (material_id % 6000))


def case_reference(material_id: int, offset: int = 0) -> dict[str, str]:
    timestamp = safe_datetime(material_id * 2 + offset)
    return {
        "case_id": f"AUTO-{timestamp.strftime('%Y%m%d-%H%M%S')}",
        "operator": OPERATORS[(material_id + offset) % len(OPERATORS)],
        "written_at": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "source_label": f"case_{timestamp.strftime('%Y%m%d_%H%M%S')}.md",
    }


def ctx(source_label: str, category: str, device: str, content: str) -> dict[str, str]:
    return {"source_label": source_label, "doc_category": category, "device_name": device, "content": content}


def manual_context(profile: Profile, unit: str, material_id: int) -> dict[str, str]:
    return ctx(
        f"{profile.code}_维护手册_{material_id % 5 + 1}.md",
        "设备说明书",
        unit,
        f"设备：{unit}\n{profile.normal_rule}\n与{profile.component}有关的候选方向包括：{profile.cause}。\n处理原则：{profile.action}。",
    )


def sop_context(profile: Profile, unit: str, material_id: int, *, stale: bool = False) -> dict[str, str]:
    revision = "2024-04-15" if stale else "2026-05-18"
    state = "该 SOP 已在 2025-12-01 被替代，不适用于当前版本设备。" if stale else "该 SOP 为当前有效版本。"
    return ctx(
        f"SOP-{profile.code}-{material_id % 9 + 1:02d}.md",
        "SOP",
        unit,
        f"修订日期：{revision}\n{state}\n{unit}异常处理流程：发现“{profile.symptom}”后，先记录参数，再{profile.action}。\n安全边界：{profile.risk}",
    )


def log_context(profile: Profile, unit: str, material_id: int, *, status: str = "candidate", symptom: str | None = None) -> dict[str, str]:
    timestamp = DATA_CUTOFF - timedelta(days=material_id % 50, minutes=(material_id * 13) % 720)
    observed_symptom = symptom or profile.symptom
    if status == "verified":
        tail = "现场已完成处理和复测，处理后关键参数恢复至控制范围，记录保留了复测结果。"
    elif status == "conflict":
        tail = "当前日志与部分历史记录不一致，且尚未完成拆检；只能形成候选判断，不能直接确认根因。"
    else:
        tail = "现场尚未完成拆检和复测，当前记录只能支持候选判断。"
    return ctx(
        f"{profile.code}_{timestamp.strftime('%Y%m%d')}_点检日志.md",
        "维保日志",
        unit,
        f"{timestamp.strftime('%Y-%m-%d')} 点检记录：{unit}出现“{observed_symptom}”。观测：{profile.evidence}。{tail}",
    )


def unconfirmed_case_context(profile: Profile, unit: str, material_id: int, *, symptom: str | None = None) -> dict[str, str]:
    return ctx(
        f"HC-{profile.code}-{material_id % 500 + 100:03d}.md",
        "案例卡",
        unit,
        f"历史案例卡：同类设备曾出现“{symptom or profile.symptom}”。记录提示优先核对{profile.component}，"
        "但该案例不是前台人工确认回写，只能作为历史参考，不能替代本次现场确认。",
    )


def reviewed_writeback_context(
    profile: Profile,
    unit: str,
    material_id: int,
    *,
    offset: int = 0,
    conflict: bool = False,
    missing: set[str] | None = None,
    truncate: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    missing = missing or set()
    reference = case_reference(material_id, offset)
    alternative = "历史结论指向入口侧阻力异常，与另一回写案例的处理方向不同。" if conflict else f"历史结论提示{profile.component}相关异常是候选方向。"
    lines = [
        f"案例编号：{reference['case_id']}",
        f"设备：{unit}",
        "文档类别：案例卡",
        REVIEWED_SOURCE,
    ]
    if "operator" not in missing:
        lines.append(f"处理人：{reference['operator']}")
    if "written_at" not in missing:
        lines.append(f"回写时间：{reference['written_at']}")
    lines.extend(["", "故障现象：", profile.symptom, "", "系统建议可能原因：", f"- {profile.cause}", "", "系统建议排查步骤：", f"1. {profile.action}", ""])
    if "actual_steps" not in missing:
        step = f"{profile.action}，随后在低风险工况下复测关键参数。"
        lines.extend(["实际处理步骤：", step[:-8] + "……" if truncate else step, ""])
    if "final_result" not in missing:
        lines.extend(["最终结论：", alternative, ""])
    if "experience_summary" not in missing:
        lines.extend(["经验总结：", "该案例仅作为历史参考，本次仍需结合当前观测和复测确认。", ""])
    lines.extend(["依据文档：", f"- {profile.code}维护手册与当班点检记录"])
    return ctx(reference["source_label"], "案例卡", unit, "\n".join(lines)), reference


def wrong_device_context(profile: Profile, material_id: int, *, same_symptom: bool = False) -> dict[str, str]:
    other = PROFILES[(PROFILES.index(profile) + 7) % len(PROFILES)]
    other_unit = unit_name(other, material_id + 900)
    symptom = profile.symptom if same_symptom else other.symptom
    return ctx(
        f"{other.code}_维修记录_{material_id % 11 + 1}.md",
        "维修记录",
        other_unit,
        f"设备：{other_unit}。记录异常：“{symptom}”。处理内容：{other.action}。"
        "该片段设备类型与当前输入不一致，不能直接作为当前设备的诊断依据。",
    )


def same_device_wrong_symptom_context(profile: Profile, unit: str, material_id: int) -> dict[str, str]:
    other = PROFILES[(PROFILES.index(profile) + 5) % len(PROFILES)]
    return ctx(
        f"{profile.code}_不同症状历史记录_{material_id % 9 + 1}.md",
        "维修记录",
        unit,
        f"设备：{unit}。历史记录的异常为“{other.symptom}”，处理内容为{other.action}。"
        f"该片段症状与当前“{profile.symptom}”不一致，不能直接迁移其结论。",
    )


def duplicate_context(source: dict[str, str]) -> dict[str, str]:
    return {**source}


def fault_payload(profile: Profile, unit: str) -> dict[str, Any]:
    return {"device": unit, "symptom": profile.symptom}


def training_payload(profile: Profile, unit: str, level: int) -> dict[str, Any]:
    summary = (
        f"当前日志提示{profile.component}可能参与异常，需按顺序排查。"
        if level >= 2
        else f"当前仅知道出现“{profile.symptom}”，缺少检索依据，不能判断具体原因。"
    )
    causes = [f"{profile.component}相关异常为候选方向"] if level >= 2 else ["证据不足，暂不列出具体原因。"]
    return {
        "device": unit,
        "symptom": profile.symptom,
        "fault_summary": summary,
        "possible_causes": causes,
        "troubleshooting_steps": [profile.action] if level >= 2 else ["补充现场点检、报警和检索资料后再制定排查步骤。"],
        "risk_notes": [profile.risk],
        "training_points": ["先区分已知事实、历史参考与待确认推断。"],
    }


def case_review_payload(profile: Profile, unit: str, material_id: int, *, verified: bool) -> dict[str, Any]:
    reference = case_reference(material_id)
    final_result = (
        f"复测记录显示，{profile.component}处理后关键参数恢复至控制范围，当前结论已由本次复测支持。"
        if verified
        else f"初步处理后症状有所缓解，但未完成完整复测；{profile.component}仅为候选方向。"
    )
    return {
        "device": unit,
        "symptom": profile.symptom,
        "actual_steps": f"{profile.action}；记录处理前后关键参数。",
        "final_result": final_result,
        "experience_summary": f"出现“{profile.symptom}”时，应先核对{profile.component}与当前运行边界。",
        "operator": reference["operator"],
        "possible_causes": [f"{profile.component}相关异常", profile.cause],
        "recommended_steps": [profile.action, "复测并完成记录。"],
        "evidence_sources": [f"{profile.code}_维护手册", f"{profile.code}_点检日志"],
    }


def base_metadata(family: str, evidence_level: int, quality: str, writeback: str, abstain: bool) -> dict[str, Any]:
    return {
        "scenario_type": family,
        "evidence_level": evidence_level,
        "retrieval_quality": quality,
        "writeback_match_type": writeback,
        "should_abstain": abstain,
        "data_cutoff": DATA_CUTOFF.strftime("%Y-%m-%d"),
    }


def make_scenario(family: str, sequence: int, split: str) -> dict[str, Any]:
    task = task_for_family(family)
    material_id = material_sequence(sequence, split)
    profile = PROFILES[material_id % len(PROFILES)]
    unit = unit_name(profile, material_id)
    contexts: list[dict[str, str]] = []
    contract: dict[str, Any] = {"writeback_policy": "none", "required_terms": []}

    if task == "fault_diagnosis":
        payload = fault_payload(profile, unit)
        if family == "fault_full_evidence":
            contexts = [manual_context(profile, unit, material_id), log_context(profile, unit, material_id), unconfirmed_case_context(profile, unit, material_id)]
            metadata = base_metadata(family, 2, "sufficient", "none", False)
        elif family == "fault_no_retrieval":
            metadata = base_metadata(family, 0, "none", "none", True)
        elif family == "fault_only_sop":
            contexts = [sop_context(profile, unit, material_id)]
            metadata = base_metadata(family, 1, "weak", "none", False)
        elif family == "fault_manual_only":
            contexts = [manual_context(profile, unit, material_id)]
            metadata = base_metadata(family, 1, "weak", "none", False)
        elif family == "fault_unconfirmed_history":
            contexts = [unconfirmed_case_context(profile, unit, material_id)]
            metadata = base_metadata(family, 3, "partial", "none", False)
        elif family == "fault_writeback_exact":
            reviewed, reference = reviewed_writeback_context(profile, unit, material_id)
            contexts = [manual_context(profile, unit, material_id), log_context(profile, unit, material_id), reviewed]
            metadata = base_metadata(family, 4, "sufficient", "exact", False)
            contract = {"writeback_policy": "exact", "references": [reference], "required_terms": ["现场复核"]}
        elif family == "fault_multi_writeback_consistent":
            first, first_ref = reviewed_writeback_context(profile, unit, material_id, offset=0)
            second, second_ref = reviewed_writeback_context(profile, unit, material_id, offset=1)
            contexts = [manual_context(profile, unit, material_id), log_context(profile, unit, material_id), first, second]
            metadata = base_metadata(family, 4, "sufficient", "multiple_consistent", False)
            contract = {"writeback_policy": "multiple_consistent", "references": [first_ref, second_ref], "required_terms": ["一致", "现场复核"]}
        elif family == "fault_multi_writeback_conflicting":
            first, first_ref = reviewed_writeback_context(profile, unit, material_id, offset=0)
            second, second_ref = reviewed_writeback_context(profile, unit, material_id, offset=1, conflict=True)
            contexts = [manual_context(profile, unit, material_id), log_context(profile, unit, material_id, status="conflict"), first, second]
            metadata = base_metadata(family, 4, "conflicting", "multiple_conflicting", False)
            contract = {"writeback_policy": "multiple_conflicting", "references": [first_ref, second_ref], "required_terms": ["冲突", "不能直接"]}
        elif family == "fault_writeback_partial":
            reviewed, reference = reviewed_writeback_context(profile, unit, material_id, missing={"actual_steps", "final_result"})
            contexts = [manual_context(profile, unit, material_id), log_context(profile, unit, material_id), reviewed]
            metadata = base_metadata(family, 3, "partial", "partial", False)
            contract = {"writeback_policy": "partial", "references": [reference], "required_terms": ["不完整"]}
        elif family == "fault_wrong_device":
            contexts = [wrong_device_context(profile, material_id)]
            metadata = base_metadata(family, 1, "noisy", "none", True)
            contract = {"writeback_policy": "none", "required_terms": ["设备不一致"]}
        elif family == "fault_same_device_wrong_symptom":
            contexts = [same_device_wrong_symptom_context(profile, unit, material_id)]
            metadata = base_metadata(family, 1, "noisy", "none", True)
            contract = {"writeback_policy": "none", "required_terms": ["症状不一致"]}
        elif family == "fault_same_symptom_wrong_device":
            contexts = [wrong_device_context(profile, material_id, same_symptom=True)]
            metadata = base_metadata(family, 1, "noisy", "none", True)
            contract = {"writeback_policy": "none", "required_terms": ["设备不一致"]}
        elif family == "fault_stale_sop":
            contexts = [sop_context(profile, unit, material_id, stale=True)]
            metadata = base_metadata(family, 1, "noisy", "none", True)
            contract = {"writeback_policy": "none", "required_terms": ["已", "不适用"]}
        elif family == "fault_duplicate_fragment":
            manual = manual_context(profile, unit, material_id)
            contexts = [manual, duplicate_context(manual), log_context(profile, unit, material_id)]
            metadata = base_metadata(family, 2, "noisy", "none", False)
            contract = {"writeback_policy": "none", "required_terms": ["重复"]}
        elif family == "fault_ordering_noise":
            contexts = [wrong_device_context(profile, material_id), manual_context(profile, unit, material_id), log_context(profile, unit, material_id)]
            metadata = base_metadata(family, 2, "noisy", "none", False)
            contract = {"writeback_policy": "none", "required_terms": ["设备不一致"]}
        elif family == "fault_truncated_marked":
            reviewed, reference = reviewed_writeback_context(profile, unit, material_id, missing={"final_result", "experience_summary"}, truncate=True)
            contexts = [manual_context(profile, unit, material_id), reviewed]
            metadata = base_metadata(family, 3, "partial", "partial", False)
            contract = {"writeback_policy": "partial", "references": [reference], "required_terms": ["截断"]}
        elif family == "fault_missing_writeback_fields":
            reviewed, reference = reviewed_writeback_context(profile, unit, material_id, missing={"operator", "written_at"})
            contexts = [manual_context(profile, unit, material_id), reviewed]
            metadata = base_metadata(family, 3, "partial", "partial", False)
            contract = {"writeback_policy": "partial", "references": [reference], "required_terms": ["缺少"]}
        elif family == "fault_missing_conclusion":
            reviewed, reference = reviewed_writeback_context(profile, unit, material_id, missing={"final_result"})
            contexts = [manual_context(profile, unit, material_id), reviewed]
            metadata = base_metadata(family, 3, "partial", "partial", False)
            contract = {"writeback_policy": "partial", "references": [reference], "required_terms": ["最终结论"]}
        elif family == "fault_missing_actual_steps":
            reviewed, reference = reviewed_writeback_context(profile, unit, material_id, missing={"actual_steps"})
            contexts = [manual_context(profile, unit, material_id), reviewed]
            metadata = base_metadata(family, 3, "partial", "partial", False)
            contract = {"writeback_policy": "partial", "references": [reference], "required_terms": ["实际处理步骤"]}
        elif family == "fault_user_retrieval_conflict":
            conflicting_symptom = PROFILES[(PROFILES.index(profile) + 2) % len(PROFILES)].symptom
            contexts = [manual_context(profile, unit, material_id), log_context(profile, unit, material_id, status="conflict", symptom=conflicting_symptom)]
            metadata = base_metadata(family, 2, "conflicting", "none", False)
            contract = {"writeback_policy": "none", "required_terms": ["冲突"]}
        else:  # pragma: no cover
            raise ValueError(f"未实现故障情景：{family}")
    elif task == "training_support":
        if family == "training_evidence_level2":
            payload = training_payload(profile, unit, 2)
            contexts = [manual_context(profile, unit, material_id), log_context(profile, unit, material_id)]
            metadata = base_metadata(family, 2, "sufficient", "none", False)
        elif family == "training_evidence_level0":
            payload = training_payload(profile, unit, 0)
            contexts = []
            metadata = base_metadata(family, 0, "none", "none", True)
        elif family == "training_conflict":
            payload = training_payload(profile, unit, 2)
            contexts = [manual_context(profile, unit, material_id), log_context(profile, unit, material_id, status="conflict")]
            metadata = base_metadata(family, 2, "conflicting", "none", False)
        else:  # pragma: no cover
            raise ValueError(f"未实现培训情景：{family}")
        contract = {"writeback_policy": "not_applicable", "required_terms": ["证据不足"] if metadata["evidence_level"] == 0 else []}
    else:
        verified = family == "case_review_verified"
        payload = case_review_payload(profile, unit, material_id, verified=verified)
        contexts = [manual_context(profile, unit, material_id), log_context(profile, unit, material_id, status="verified" if verified else "candidate")]
        metadata = base_metadata(family, 5 if verified else 2, "sufficient" if verified else "partial", "not_applicable", False)
        contract = {"writeback_policy": "not_applicable", "required_terms": [] if verified else ["候选"]}

    return {
        "id": f"RT3-{split}-{family}-{sequence:04d}",
        "task": task,
        "split": split,
        "payload": payload,
        "context_items": contexts,
        "metadata": metadata,
        "contract": contract,
    }


def build_scenarios(matrix: dict[str, Any], split: str, limit: int | None = None, one_per_family: bool = False) -> list[dict[str, Any]]:
    counts = matrix.get(split)
    if not isinstance(counts, dict):
        raise ValueError(f"情景矩阵缺少 split={split}。")
    if one_per_family:
        return [make_scenario(family, index, split) for index, family in enumerate(counts)]
    scenarios: list[dict[str, Any]] = []
    sequence = 0
    for family, count in counts.items():
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"情景配额错误：{family}={count!r}")
        for _ in range(count):
            scenarios.append(make_scenario(family, sequence, split))
            sequence += 1
            if limit and len(scenarios) >= limit:
                return scenarios
    return scenarios


def contract_instruction(scenario: dict[str, Any]) -> str:
    metadata = scenario["metadata"]
    contract = scenario["contract"]
    lines = [
        f"证据等级={metadata['evidence_level']}，检索质量={metadata['retrieval_quality']}，是否应拒绝具体诊断={metadata['should_abstain']}。",
        "若证据等级低于 5，summary、possible_causes、root_cause_summary 和 review_note 不得把候选写成已确认根因；使用“提示、候选、尚不能确认、需复核”等表达。禁止写“A 导致/引起 B”，即使前面带有“候选”，应改写为“A 可能参与或与 B 相关”。",
        "assessment_checklist 为纯字符串数组，不得使用 □、☐ 等勾选符号。",
    ]
    if metadata["evidence_level"] == 0:
        lines.append("无检索：possible_causes 必须只包含“证据不足，暂不列出具体原因。”；只建议补充证据和安全边界。")
    if metadata["should_abstain"]:
        lines.append("当前片段无关、过期或设备/症状不一致，不能据此展开任何具体专业原因；possible_causes 必须只包含“证据不足，暂不列出具体原因。”。")
    policy = contract.get("writeback_policy")
    refs = contract.get("references", [])
    if policy in {"exact", "multiple_consistent", "multiple_conflicting"}:
        ref_text = "；".join(f"{ref['case_id']} / {ref['operator']} / {ref['written_at']}" for ref in refs)
        lines.append(f"完整回写引用必须包含：{ref_text}。历史案例仍不能直接替代本次现场确认。")
    elif policy == "partial":
        lines.append("这是带人工回写来源标记但字段不完整或被截断的片段；matched_writeback_case_note 必须说明“疑似/不完整回写参考”及缺失项，不能确认本次根因。")
    elif policy == "none":
        lines.append("matched_writeback_case_note 必须为空字符串。")
    if contract.get("required_terms"):
        lines.append(f"输出必须明确体现这些判断词或事实：{'、'.join(contract['required_terms'])}。")
    return "\n".join(lines)


def teacher_messages(scenario: dict[str, Any]) -> list[dict[str, str]]:
    task = scenario["task"]
    runtime_system = build_runtime_system_prompt(task)
    runtime_user = build_runtime_user_message(task, scenario["payload"], scenario["context_items"])
    return [
        {
            "role": "system",
            "content": "你是工业维保 RAG 金标准标注教师。可以在内部推理，但最终只输出 assistant_output JSON，绝不输出思考过程、解释或额外字段。",
        },
        {
            "role": "user",
            "content": (
                "请根据线上请求生成唯一的目标 assistant JSON。只能使用输入字段和检索片段中的事实；"
                "必须遵守证据等级约束。\n\n"
                f"[线上 system]\n{runtime_system}\n\n"
                f"[线上 user]\n{runtime_user}\n\n"
                f"[v3 验收契约]\n{contract_instruction(scenario)}\n\n"
                "返回：{\"assistant_output\": { ... }}"
            ),
        },
    ]


def generate_target(
    client: OpenAI,
    scenario: dict[str, Any],
    model: str,
    max_tokens: int,
    *,
    previous_output: dict[str, Any] | None = None,
    correction_errors: list[str] | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            messages = teacher_messages(scenario)
            if previous_output is not None and correction_errors:
                messages[-1]["content"] += (
                    "\n\n[上一版输出]\n"
                    f"{json.dumps(previous_output, ensure_ascii=False)}\n"
                    "[自动校验错误]\n- "
                    + "\n- ".join(correction_errors)
                    + "\n请修正后仅返回 assistant_output。"
                    "低证据等级绝不能使用“导致”或“引起”的直接因果句；改为“可能参与”“与当前症状相关”。"
                    "若契约要求重复、冲突、设备不一致等判断词，必须在 evidence_observations 中明确写出该判断。"
                )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "enabled"}, "user_id": "zhiweishi_runtime_sft_v3"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            if isinstance(payload, dict) and isinstance(payload.get("assistant_output"), dict):
                return payload["assistant_output"]
            if isinstance(payload, dict) and set(payload) == set(TASK_SCHEMAS[scenario["task"]]):
                return payload
            raise ValueError("教师响应未返回 assistant_output。")
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"教师调用连续失败：{last_error}")


def contains_illegal_strong_term(text: str) -> bool:
    cleaned = text.replace("可能导致", "").replace("可能会导致", "").replace("不代表导致", "")
    return "导致" in cleaned or any(term in cleaned for term in STRONG_TERMS)


def answer_text(answer: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in answer.values():
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
    return "\n".join(parts)


def validate_v3_contract(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    example_id = str(record.get("id", "unknown"))
    scenario = record.get("scenario")
    messages = record.get("messages")
    task = record.get("task")
    if not isinstance(scenario, dict) or not isinstance(messages, list) or len(messages) != 3 or task not in TASK_SCHEMAS:
        return [f"{example_id}: 缺少 v3 运行时审计信息。"]
    payload = scenario.get("payload")
    contexts = scenario.get("context_items")
    metadata = scenario.get("metadata")
    contract = scenario.get("contract")
    if not isinstance(payload, dict) or not isinstance(contexts, list) or not isinstance(metadata, dict) or not isinstance(contract, dict):
        return [f"{example_id}: v3 scenario 字段不完整。"]
    required_metadata = {"scenario_type", "evidence_level", "retrieval_quality", "writeback_match_type", "should_abstain", "data_cutoff"}
    if set(metadata) != required_metadata:
        errors.append(f"{example_id}: 审计元数据键不完整。")
    if messages[0].get("content") != build_runtime_system_prompt(task):
        errors.append(f"{example_id}: system 与线上构造函数不一致。")
    if messages[1].get("content") != build_runtime_user_message(task, payload, contexts):
        errors.append(f"{example_id}: user 与线上构造函数不一致。")
    try:
        answer = json.loads(str(messages[2].get("content", "")))
    except json.JSONDecodeError:
        return errors
    full_text = answer_text(answer)
    if "□" in full_text or "☐" in full_text:
        errors.append(f"{example_id}: 输出不得包含 checklist 符号。")
    for date_text in DATE_PATTERN.findall(messages[1].get("content", "") + "\n" + full_text):
        if datetime.strptime(date_text, "%Y-%m-%d") > DATA_CUTOFF:
            errors.append(f"{example_id}: 出现晚于 cutoff 的日期 {date_text}。")
    level = metadata.get("evidence_level")
    if task == "fault_diagnosis":
        note = str(answer.get("matched_writeback_case_note", "")).strip()
        policy = contract.get("writeback_policy")
        if policy == "none" and note:
            errors.append(f"{example_id}: 非回写命中场景的 matched_writeback_case_note 必须为空。")
        if policy in {"exact", "multiple_consistent", "multiple_conflicting"}:
            for ref in contract.get("references", []):
                for key in ("case_id", "operator", "written_at"):
                    if str(ref[key]) not in note:
                        errors.append(f"{example_id}: 回写说明缺少 {key}={ref[key]}。")
        if policy == "partial":
            if not note or not any(term in note for term in ("疑似", "不完整", "缺少", "截断")):
                errors.append(f"{example_id}: 不完整回写必须在 note 中明确降级。")
        if level == 0 or metadata.get("should_abstain"):
            causes = answer.get("possible_causes")
            if causes != ["证据不足，暂不列出具体原因。"]:
                errors.append(f"{example_id}: 无可靠证据时不得展开具体原因。")
        if isinstance(level, int) and level < 5:
            candidate_text = "\n".join(
                [
                    str(answer.get("summary", "")),
                    *[str(item) for item in answer.get("possible_causes", [])],
                ]
            )
            if contains_illegal_strong_term(candidate_text):
                errors.append(f"{example_id}: 低证据等级出现过强因果结论。")
            uncertainty = str(answer.get("uncertainty_note", ""))
            if not any(term in uncertainty for term in ("尚不能确认", "需", "证据不足", "待", "不确定", "当前", "无法", "不能", "无任何")):
                errors.append(f"{example_id}: 低证据等级缺少有效不确定性边界。")
    if task == "case_review" and isinstance(level, int) and level < 5:
        root = str(answer.get("root_cause_summary", ""))
        if contains_illegal_strong_term(root):
            errors.append(f"{example_id}: 候选复盘不得写确定性根因。")
        if "候选" not in str(answer.get("review_note", "")):
            errors.append(f"{example_id}: 候选复盘 review_note 必须标明候选状态。")
    required_terms = contract.get("required_terms", [])
    missing_terms = []
    for term in required_terms:
        alternatives = TERM_ALTERNATIVES.get(term, (term,))
        if not any(candidate in full_text for candidate in alternatives):
            missing_terms.append(term)
    if missing_terms:
        errors.append(f"{example_id}: 未体现要求的判断词或事实：{missing_terms}。")
    return errors


def build_record(scenario: dict[str, Any], assistant_output: dict[str, Any], model: str) -> tuple[dict[str, Any] | None, list[str]]:
    task = scenario["task"]
    messages = [
        {"role": "system", "content": build_runtime_system_prompt(task)},
        {"role": "user", "content": build_runtime_user_message(task, scenario["payload"], scenario["context_items"])},
        {"role": "assistant", "content": json.dumps(assistant_output, ensure_ascii=False)},
    ]
    record = {
        "id": scenario["id"],
        "task": task,
        "split": scenario["split"],
        "review_status": "teacher_generated",
        "source_mode": "synthetic_runtime_v3",
        "messages": messages,
        "scenario": {
            "payload": scenario["payload"],
            "context_items": scenario["context_items"],
            "metadata": scenario["metadata"],
            "contract": scenario["contract"],
        },
        "teacher": {"model": model, "thinking": "enabled", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }
    errors = validate_messages(messages, record["id"])
    errors.extend(validate_assistant_output(task, messages[2]["content"], record["id"]))
    errors.extend(validate_runtime_semantics(task, messages, record["id"]))
    errors.extend(validate_v3_contract(record))
    return (record if not errors else None), errors


def generate_record(client: OpenAI, scenario: dict[str, Any], model: str, max_tokens: int) -> tuple[dict[str, Any] | None, list[str], dict[str, Any] | None]:
    previous: dict[str, Any] | None = None
    errors: list[str] = []
    for _ in range(3):
        output = generate_target(client, scenario, model, max_tokens, previous_output=previous, correction_errors=errors or None)
        record, errors = build_record(scenario, output, model)
        if record is not None:
            return record, [], output
        previous = output
    return None, errors, previous


def coverage_report(records: list[dict[str, Any]], planned: list[dict[str, Any]]) -> dict[str, Any]:
    def count(values: list[str]) -> dict[str, int]:
        output: dict[str, int] = {}
        for value in values:
            output[value] = output.get(value, 0) + 1
        return dict(sorted(output.items()))

    metadata = [record["scenario"]["metadata"] for record in records]
    return {
        "planned": count([scenario["metadata"]["scenario_type"] for scenario in planned]),
        "accepted": {
            "total": len(records),
            "by_task": count([record["task"] for record in records]),
            "by_scenario_type": count([item["scenario_type"] for item in metadata]),
            "by_evidence_level": count([str(item["evidence_level"]) for item in metadata]),
            "by_retrieval_quality": count([item["retrieval_quality"] for item in metadata]),
            "by_writeback_match_type": count([item["writeback_match_type"] for item in metadata]),
            "should_abstain": sum(bool(item["should_abstain"]) for item in metadata),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="构建运行时同构、证据约束的 v3 SFT 数据。")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--split", choices=tuple(SPLIT_OFFSETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--one-per-family", action="store_true")
    parser.add_argument("--repair-rejected", type=Path, help="从此前的 *_rejected.jsonl 读取原始情景并重新标注。")
    parser.add_argument("--existing-dataset", type=Path, help="修复模式下保留的既有数据集；将与新通过样本合并写到 --output。")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.max_tokens <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("workers、max-tokens 和 limit 必须为正数。")
    if args.repair_rejected and (args.one_per_family or args.limit is not None):
        parser.error("--repair-rejected 不能与 --one-per-family 或 --limit 同时使用。")
    if args.one_per_family and args.limit is not None:
        parser.error("--one-per-family 与 --limit 不能同时使用。")
    if args.repair_rejected:
        rejected_rows = load_jsonl(args.repair_rejected.resolve())
        scenarios = [row["scenario"] for row in rejected_rows if isinstance(row.get("scenario"), dict)]
        if not scenarios:
            parser.error("修复文件中没有可用的 scenario。")
        mismatched_splits = {str(scenario.get("split")) for scenario in scenarios} - {args.split}
        if mismatched_splits:
            parser.error(f"修复文件的 split 与 --split 不一致：{sorted(mismatched_splits)}")
    else:
        matrix = load_json(args.matrix.resolve())
        scenarios = build_scenarios(matrix, args.split, args.limit, args.one_per_family)
    planned_counts = coverage_report([], scenarios)["planned"]
    print(f"v3 split={args.split}；情景数={len(scenarios)}；并发={args.workers}；情景分布={planned_counts}")
    if not args.execute:
        print("预览模式：不调用教师 API。线上格式、证据等级和 hard negative 均会由本地构造器校验。")
        return 0
    config = AppConfig.from_env()
    if not config.deepseek_api_key:
        print("未配置 deepseek_api_key，无法调用教师模型。")
        return 1
    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        print(f"输出已存在，拒绝覆盖：{output_path}")
        return 1
    client = OpenAI(api_key=config.deepseek_api_key, base_url=config.deepseek_base_url)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_record, client, scenario, args.model, args.max_tokens): scenario for scenario in scenarios}
        for future in as_completed(futures):
            scenario = futures[future]
            completed += 1
            try:
                record, errors, raw = future.result()
                if record is None:
                    rejected.append({"scenario": scenario, "reason": errors, "raw": raw})
                else:
                    accepted.append(record)
            except Exception as error:
                rejected.append({"scenario": scenario, "reason": [f"教师调用失败：{error}"]})
            if completed % 20 == 0 or completed == len(scenarios):
                print(f"进度 {completed}/{len(scenarios)}；接受 {len(accepted)}；拒绝 {len(rejected)}。")
    existing: list[dict[str, Any]] = []
    if args.existing_dataset:
        existing_path = args.existing_dataset.resolve()
        if not existing_path.exists():
            print(f"既有数据集不存在：{existing_path}")
            return 1
        existing = load_jsonl(existing_path)
    combined = [*existing, *accepted]
    ids = [str(item.get("id", "")) for item in combined]
    if len(ids) != len(set(ids)):
        print("合并后存在重复样本 ID，拒绝写入。")
        return 1
    combined.sort(key=lambda item: item["id"])
    write_jsonl(output_path, combined, overwrite=args.overwrite)
    if rejected:
        write_jsonl(output_path.with_name(f"{output_path.stem}_rejected.jsonl"), rejected, overwrite=args.overwrite)
    output_path.with_name(f"{output_path.stem}_coverage_report.json").write_text(
        json.dumps(coverage_report(combined, [*[{"metadata": item["scenario"]["metadata"]} for item in existing], *scenarios]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"完成：接受 {len(accepted)} 条；拒绝 {len(rejected)} 条；输出={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
