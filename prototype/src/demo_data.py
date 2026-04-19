from __future__ import annotations


DEMO_CASES = [
    {
        "device": "液压泵站",
        "symptom": "压力波动且伴随异响",
        "possible_causes": [
            "液压油不足或油液污染",
            "吸油管路进气或接头松动",
            "滤芯堵塞导致供油不稳定",
            "泵体磨损导致内部泄漏",
        ],
        "troubleshooting_steps": [
            "先确认油位、油温和油液清洁度是否异常",
            "检查吸油管路和接头是否存在漏气或松动",
            "查看滤芯状态并确认是否堵塞",
            "在低负载状态下听诊泵体异响并记录异常位置",
            "如仍无法定位，停止继续高负荷运行并申请进一步检修",
        ],
        "risk_notes": [
            "继续高负荷运行可能加剧泵体磨损",
            "若伴随明显温升，应避免长时间连续运行",
        ],
        "evidence_sources": [
            "设备说明书 第3章 液压系统维护",
            "SOP-07 液压系统点检流程",
            "历史案例 HC-002 压力异常排查记录",
        ],
        "escalation_advice": "若完成基础排查后仍有异响和波动，建议停机并上报设备管理员。",
        "training_points": [
            "排查液压问题时优先看油位、油温、油质",
            "异响与压力波动常常同时出现，应先排除供油不稳定因素",
            "发现持续异响时不要直接高负荷试运行",
        ],
    },
    {
        "device": "数控主轴单元",
        "symptom": "振动异常且加工精度下降",
        "possible_causes": [
            "主轴轴承磨损",
            "刀具安装偏心",
            "联轴器松动",
            "主轴冷却状态异常",
        ],
        "troubleshooting_steps": [
            "检查刀具夹持和安装同心度",
            "观察主轴温度和冷却循环状态",
            "检查联轴器、紧固件和连接部位是否松动",
            "结合历史维护记录判断是否存在轴承老化趋势",
            "必要时暂停加工并安排精密校验",
        ],
        "risk_notes": [
            "持续振动可能扩大加工误差并损伤主轴轴承",
            "强行继续加工可能造成工件批量报废",
        ],
        "evidence_sources": [
            "设备说明书 第5章 主轴维护规范",
            "SOP-12 主轴异常检查流程",
            "历史案例 NC-004 主轴振动复盘记录",
        ],
        "escalation_advice": "若振动幅度持续增大，建议暂停加工任务并转交专业检修人员。",
        "training_points": [
            "主轴振动问题要先区分刀具因素和本体因素",
            "精度下降时需同步检查机械状态与冷却状态",
            "涉及精密部件时优先做保护性停机",
        ],
    },
]


def get_devices() -> list[str]:
    return sorted({item["device"] for item in DEMO_CASES})


def get_symptoms(device: str) -> list[str]:
    return [item["symptom"] for item in DEMO_CASES if item["device"] == device]


def find_case(device: str, symptom: str) -> dict | None:
    for item in DEMO_CASES:
        if item["device"] == device and item["symptom"] == symptom:
            return item
    return None


def build_training_questions(case: dict) -> list[str]:
    return [
        f"遇到“{case['symptom']}”时，第一步建议检查什么？",
        f"该故障继续运行最主要的风险是什么？",
        f"从哪些文档可以找到与“{case['device']}”相关的排查依据？",
        f"什么情况下需要停机并上报？",
        f"本案例最值得沉淀为新人培训要点的经验是什么？",
    ]
