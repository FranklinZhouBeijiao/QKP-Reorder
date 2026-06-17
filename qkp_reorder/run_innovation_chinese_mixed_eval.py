from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qkp_reorder.compressors import ContextCompressor
from qkp_reorder.llm_inference import generate_answer, load_project_env
from qkp_reorder.ours_utils import (
    compute_protection_bonus,
    extract_question_keywords,
    is_jieba_available,
    split_into_segments,
    tokenize_for_bm25,
)
from qkp_reorder.prompt_builder import build_qa_prompt
from qkp_reorder.tokenizer_utils import count_tokens


RESULT_DIR = PROJECT_ROOT / "results" / "innovation"
LOG_DIR = PROJECT_ROOT / "logs" / "innovation"
DATA_DIR = PROJECT_ROOT / "data" / "innovation"
DOC_DIR = PROJECT_ROOT.parent / "innovation_docs"

DATASET_OUT = DATA_DIR / "qkp_chinese_mixed_dataset.jsonl"
RESULT_OUT = RESULT_DIR / "qkp_chinese_mixed_results.csv"
SUMMARY_OUT = RESULT_DIR / "qkp_chinese_mixed_summary.csv"
API_RESULT_OUT = RESULT_DIR / "qkp_chinese_mixed_api_results.csv"
API_SUMMARY_OUT = RESULT_DIR / "qkp_chinese_mixed_api_summary.csv"
CASE_STUDY_OUT = RESULT_DIR / "qkp_chinese_mixed_case_study.md"
LOG_OUT = LOG_DIR / "qkp_chinese_mixed_log.txt"
DOC_OUT = DOC_DIR / "阶段9_QKP-Reorder中文中英混合验证执行文档.md"

METHODS = [
    "original",
    "truncate",
    "bm25",
    "llmlingua",
    "longllmlingua",
    "ours_bm25_only",
    "ours_keyword",
    "ours_full",
]
METHOD_LABELS = {
    "original": "Original",
    "truncate": "Truncate",
    "bm25": "BM25",
    "llmlingua": "LLMLingua",
    "longllmlingua": "LongLLMLingua",
    "ours_bm25_only": "QKP-Reorder-BM25",
    "ours_keyword": "QKP-Reorder-Keyword",
    "ours_full": "QKP-Reorder",
}
METHOD_RANK = {method: idx for idx, method in enumerate(METHODS)}
API_SAMPLE_COUNT = 10
API_METHODS = ["ours_full"]
TOKEN_BUDGET = 60
API_RESULT_COLUMNS = [
    "sample_id",
    "language_type",
    "method",
    "method_label",
    "question",
    "answers",
    "prediction",
    "api_contains_answer",
    "empty_prediction",
    "error_message",
    "answer_time",
    "api_total_tokens",
    "api_prompt_tokens",
    "api_completion_tokens",
    "retries",
]
API_SUMMARY_COLUMNS = [
    "method",
    "method_label",
    "language_type",
    "rows",
    "api_contains_answer",
    "empty_predictions",
    "errors",
    "avg_answer_time",
    "avg_api_total_tokens",
]


def _build_context(noise: list[str], evidence: str, tail: list[str]) -> str:
    return "\n".join(noise + [evidence] + tail)


CHINESE_MIXED_DATASET: list[dict[str, Any]] = [
    {
        "sample_id": "cm_001",
        "language_type": "zh",
        "question": "2024年北京人工智能大会上，哪家公司发布了DeepSeek-R1相关工具？",
        "answers": ["深度求索", "DeepSeek"],
        "keywords": ["2024", "北京", "人工智能大会", "DeepSeek-R1"],
        "context": _build_context(
            [
                "上海团队介绍了多模态检索系统，主要面向医学影像。",
                "广州论坛讨论了低空经济数据平台，未涉及大模型工具。",
                "南京高校展示了校园智能问答助手，发布时间是2023年。",
                "成都实验室报告了中文语音识别基准，样本数为1,234。",
            ],
            "2024年北京人工智能大会上，深度求索发布了DeepSeek-R1相关工具，重点展示中文推理能力。",
            [
                "随后NASA报告了AI_report-2024中的英文实验，增长12.5%。",
                "闭幕环节讨论了开源生态和模型安全。",
            ],
        ),
    },
    {
        "sample_id": "cm_002",
        "language_type": "zh",
        "question": "杭州绿色算力论坛中，承担冷却系统优化的是哪个团队？",
        "answers": ["青岚实验室"],
        "keywords": ["杭州", "绿色算力", "冷却系统", "优化"],
        "context": _build_context(
            [
                "北京会议先介绍了数据治理平台，主题集中在政务问答。",
                "深圳团队发布了边缘计算盒子，强调低功耗部署。",
                "合肥高校报告了量子教学工具，未涉及机房冷却。",
            ],
            "在杭州绿色算力论坛中，青岚实验室承担冷却系统优化任务，并把能耗降低到原方案的82%。",
            [
                "论坛最后讨论了碳交易数据接口和园区级调度。",
                "媒体报道还提到一个名为RiverStack的监控脚本。",
            ],
        ),
    },
    {
        "sample_id": "cm_003",
        "language_type": "zh",
        "question": "苏州智能制造展上用于缺陷检测的模型叫什么？",
        "answers": ["瑶光-Inspect"],
        "keywords": ["苏州", "智能制造", "缺陷检测", "模型"],
        "context": _build_context(
            [
                "天津展区展示了仓储机器人，关注点是路径规划。",
                "武汉团队介绍了安全帽识别系统，部署在建筑工地。",
                "重庆论坛讨论了工业知识图谱，重点是设备标签。",
            ],
            "苏州智能制造展上，星瀚科技发布了用于缺陷检测的模型瑶光-Inspect，支持中文质检报告生成。",
            [
                "后续报告比较了两条产线的良率变化。",
                "评委还提到模型压缩会影响边缘设备延迟。",
            ],
        ),
    },
    {
        "sample_id": "cm_004",
        "language_type": "zh",
        "question": "在广州城市交通实验中，预测拥堵指数的系统名称是什么？",
        "answers": ["穗行Brain"],
        "keywords": ["广州", "城市交通", "拥堵指数", "系统"],
        "context": _build_context(
            [
                "西安报告介绍了历史街区客流预测，使用传统统计模型。",
                "上海案例关注停车场支付接口，没有预测拥堵指数。",
                "青岛团队展示了港口物流看板，指标是吞吐量。",
            ],
            "在广州城市交通实验中，市交委使用穗行Brain预测拥堵指数，并把早高峰预警提前了18分钟。",
            [
                "实验还记录了公交到站误差和出租车订单密度。",
                "总结部分强调数据隐私和道路传感器维护。",
            ],
        ),
    },
    {
        "sample_id": "cm_005",
        "language_type": "zh",
        "question": "南京数字文旅项目中，负责古籍问答模块的是哪个公司？",
        "answers": ["墨衡科技"],
        "keywords": ["南京", "数字文旅", "古籍问答", "公司"],
        "context": _build_context(
            [
                "杭州项目展示了AR导览，合作方是本地博物馆。",
                "长沙项目使用游客画像系统，主要服务夜间经济。",
                "福州案例讲解了非遗视频检索，没有古籍问答模块。",
            ],
            "南京数字文旅项目中，墨衡科技负责古籍问答模块，系统能根据繁体文本回答人物关系问题。",
            [
                "项目验收报告列出了30个景区的接入计划。",
                "另一个团队负责票务预测和人流热力图。",
            ],
        ),
    },
    {
        "sample_id": "cm_006",
        "language_type": "zh",
        "question": "深圳医疗AI沙龙中，哪个模型用于病理报告摘要？",
        "answers": ["杏林-Summary"],
        "keywords": ["深圳", "医疗AI", "病理报告", "摘要"],
        "context": _build_context(
            [
                "北京医院分享了门诊分诊机器人，重点是多轮对话。",
                "厦门团队介绍了药品说明书检索系统。",
                "成都公司展示了医保审核流程自动化工具。",
            ],
            "深圳医疗AI沙龙中，南山医工联合体使用杏林-Summary生成病理报告摘要，并保留关键诊断术语。",
            [
                "沙龙最后讨论了医疗数据脱敏和模型审计。",
                "主持人提醒所有结果不能替代医生诊断。",
            ],
        ),
    },
    {
        "sample_id": "cm_007",
        "language_type": "zh",
        "question": "成都教育技术试点中，自动批改作文的工具名称是什么？",
        "answers": ["蜀学批阅"],
        "keywords": ["成都", "教育技术", "自动批改", "作文"],
        "context": _build_context(
            [
                "上海学校试点了口语测评，面向初中英语。",
                "南京平台统计了数学错题本，重点是知识点标签。",
                "武汉团队讨论了作业拍照识别，但没有作文批改。",
            ],
            "成都教育技术试点中，市教科院引入蜀学批阅自动批改作文，并给出结构、论据和语言三类反馈。",
            [
                "试点覆盖12所学校和4,800名学生。",
                "后续计划加入教师复核入口。",
            ],
        ),
    },
    {
        "sample_id": "cm_008",
        "language_type": "zh",
        "question": "重庆工业互联网会议中，哪套平台监测设备异常？",
        "answers": ["山城IoT-Guard"],
        "keywords": ["重庆", "工业互联网", "设备异常", "平台"],
        "context": _build_context(
            [
                "杭州企业介绍了供应链金融模块，指标是回款周期。",
                "天津工厂展示了数字孪生车间，重点是三维可视化。",
                "长沙团队讨论了安全培训系统。",
            ],
            "重庆工业互联网会议中，山城IoT-Guard用于监测设备异常，并在压缩机温度升高时触发告警。",
            [
                "会议材料还列出传感器校准周期和维护责任人。",
                "专家建议把异常日志接入知识库。",
            ],
        ),
    },
    {
        "sample_id": "cm_009",
        "language_type": "zh",
        "question": "青岛海洋数据项目中，预测赤潮风险的模型是谁发布的？",
        "answers": ["海瞳实验室"],
        "keywords": ["青岛", "海洋数据", "赤潮风险", "模型"],
        "context": _build_context(
            [
                "宁波港口项目关注集装箱识别，使用视频分析。",
                "厦门海岸线项目统计游客密度，没有赤潮预测。",
                "大连研究组介绍了船舶轨迹聚类。",
            ],
            "青岛海洋数据项目中，海瞳实验室发布了预测赤潮风险的模型，并融合水温、盐度和叶绿素指标。",
            [
                "项目计划每6小时更新一次风险地图。",
                "附录中列出数据接口和传感器编号。",
            ],
        ),
    },
    {
        "sample_id": "cm_010",
        "language_type": "zh",
        "question": "合肥科研助手评测中，哪一个系统负责论文公式检索？",
        "answers": ["科枢-Formula"],
        "keywords": ["合肥", "科研助手", "论文公式", "检索"],
        "context": _build_context(
            [
                "北京团队测试了文献摘要生成，关注引用完整性。",
                "上海高校评估了图表问答系统。",
                "南京实验讨论了专利分类，没有处理论文公式。",
            ],
            "合肥科研助手评测中，科枢-Formula负责论文公式检索，可以根据中文描述定位LaTeX表达式。",
            [
                "评测集包含物理、数学和机器学习三类论文。",
                "报告还统计了检索延迟和缓存命中率。",
            ],
        ),
    },
    {
        "sample_id": "cm_011",
        "language_type": "zh",
        "question": "长沙政务问答系统中，负责政策时效检查的模块叫什么？",
        "answers": ["星政-Validity"],
        "keywords": ["长沙", "政务问答", "政策时效", "模块"],
        "context": _build_context(
            [
                "广州政务平台展示了办事材料推荐。",
                "福州团队讨论了热线工单归并。",
                "武汉系统支持地图检索，但不检查政策时效。",
            ],
            "长沙政务问答系统中，星政-Validity负责政策时效检查，能够标记已废止文件和仍有效条款。",
            [
                "系统上线前进行了三轮人工审核。",
                "运维团队每天同步一次法规库。",
            ],
        ),
    },
    {
        "sample_id": "cm_012",
        "language_type": "zh",
        "question": "西安城市安全项目中，识别燃气泄漏风险的模型叫什么？",
        "answers": ["秦安GasNet"],
        "keywords": ["西安", "城市安全", "燃气泄漏", "模型"],
        "context": _build_context(
            [
                "重庆项目分析了桥梁震动传感器数据。",
                "成都系统识别电梯困人事件。",
                "郑州平台关注暴雨积水预警。",
            ],
            "西安城市安全项目中，秦安GasNet识别燃气泄漏风险，并结合报警记录和管网压力数据。",
            [
                "项目要求高风险点位在5分钟内通知值班人员。",
                "后续还会接入社区巡检记录。",
            ],
        ),
    },
    {
        "sample_id": "cm_013",
        "language_type": "zh",
        "question": "武汉农业智能体项目中，判断水稻病害的工具名称是什么？",
        "answers": ["稻知Doctor"],
        "keywords": ["武汉", "农业智能体", "水稻病害", "工具"],
        "context": _build_context(
            [
                "哈尔滨项目监测大豆长势，使用无人机图像。",
                "昆明团队介绍了花卉价格预测。",
                "兰州平台提供灌溉建议，但不识别病害。",
            ],
            "武汉农业智能体项目中，稻知Doctor用于判断水稻病害，并能给出纹枯病和稻瘟病的防治建议。",
            [
                "项目采集了2,600张田间图片。",
                "专家要求所有建议附带置信度。",
            ],
        ),
    },
    {
        "sample_id": "cm_014",
        "language_type": "zh",
        "question": "天津港口调度实验中，哪个算法用于泊位分配？",
        "answers": ["津港Opt-27"],
        "keywords": ["天津", "港口调度", "泊位分配", "算法"],
        "context": _build_context(
            [
                "青岛报告了冷链货物温控系统。",
                "宁波团队展示了海关单证识别。",
                "上海港区讨论了无人集卡路线。",
            ],
            "天津港口调度实验中，津港Opt-27用于泊位分配，使平均等待时间下降9.8%。",
            [
                "实验还记录了潮汐、船长和装卸窗口。",
                "调度员保留人工确认权限。",
            ],
        ),
    },
    {
        "sample_id": "cm_015",
        "language_type": "zh",
        "question": "厦门金融风控试点中，识别异常转账的系统名称是什么？",
        "answers": ["鹭盾Risk"],
        "keywords": ["厦门", "金融风控", "异常转账", "系统"],
        "context": _build_context(
            [
                "上海银行测试了智能客服摘要。",
                "深圳平台评估了反洗钱规则库。",
                "北京团队关注贷款申请材料核验。",
            ],
            "厦门金融风控试点中，鹭盾Risk识别异常转账，并在交易金额超过历史均值300%时触发复核。",
            [
                "试点阶段只输出风险提示，不自动冻结账户。",
                "审计日志保留90天。",
            ],
        ),
    },
    {
        "sample_id": "cm_016",
        "language_type": "mixed",
        "question": "In the Beijing AI demo, which company released the DeepSeek-R1 helper?",
        "answers": ["DeepSeek", "深度求索"],
        "keywords": ["Beijing", "DeepSeek-R1", "helper", "released"],
        "context": _build_context(
            [
                "Shanghai Lab showed a multilingual search demo for finance documents.",
                "广州团队介绍了边缘部署方案，主题不是推理工具。",
                "Nanjing University discussed course QA over Chinese textbooks.",
            ],
            "At the 2024 Beijing AI demo, DeepSeek released the DeepSeek-R1 helper for Chinese reasoning workflows.",
            [
                "The next talk compared token budgets across English and Chinese inputs.",
                "会议记录还提到样本数为2,048。",
            ],
        ),
    },
    {
        "sample_id": "cm_017",
        "language_type": "mixed",
        "question": "哪个系统在 Hangzhou Cloud Expo 负责 bilingual log analysis？",
        "answers": ["LogBridge-CN"],
        "keywords": ["Hangzhou", "Cloud Expo", "bilingual", "log analysis"],
        "context": _build_context(
            [
                "北京展区展示了GPU资源看板。",
                "Shenzhen team introduced a low-latency vector cache.",
                "成都团队讨论了中文客服摘要。",
            ],
            "At Hangzhou Cloud Expo, LogBridge-CN handled bilingual log analysis and aligned English error codes with Chinese incident notes.",
            [
                "The report also mentioned a 12.5% drop in manual triage time.",
                "闭幕演讲关注云原生成本优化。",
            ],
        ),
    },
    {
        "sample_id": "cm_018",
        "language_type": "mixed",
        "question": "What tool translated policy clauses in the Nanjing GovTech pilot?",
        "answers": ["ClauseMix-Translator"],
        "keywords": ["Nanjing", "GovTech", "policy clauses", "translated"],
        "context": _build_context(
            [
                "Wuhan demo focused on map-based service search.",
                "深圳项目处理热线工单，没有翻译政策条款。",
                "Shanghai team tested OCR for business licenses.",
            ],
            "In the Nanjing GovTech pilot, ClauseMix-Translator translated policy clauses between Chinese and English for foreign applicants.",
            [
                "The pilot stored all revision history for audit.",
                "项目组计划每周同步法规库。",
            ],
        ),
    },
    {
        "sample_id": "cm_019",
        "language_type": "mixed",
        "question": "广州 smart traffic test 中预测 bus delay 的模型是什么？",
        "answers": ["CantonDelayNet"],
        "keywords": ["广州", "smart traffic", "bus delay", "模型"],
        "context": _build_context(
            [
                "Chengdu system predicted subway passenger flow.",
                "北京交通项目分析停车场空位。",
                "Qingdao dashboard tracked port container movement.",
            ],
            "In the 广州 smart traffic test, CantonDelayNet predicted bus delay and used Chinese stop names plus English weather feeds.",
            [
                "The average alert came 14 minutes before the delay.",
                "后续版本会接入出租车订单数据。",
            ],
        ),
    },
    {
        "sample_id": "cm_020",
        "language_type": "mixed",
        "question": "Which model summarized Chinese pathology reports in the Shenzhen clinic trial?",
        "answers": ["MedBrief-ZH"],
        "keywords": ["Shenzhen", "clinic", "Chinese pathology", "summarized"],
        "context": _build_context(
            [
                "Hangzhou hospital tested appointment routing.",
                "北京团队评估了医保审核助手。",
                "Xiamen researchers discussed drug instruction retrieval.",
            ],
            "In the Shenzhen clinic trial, MedBrief-ZH summarized Chinese pathology reports and preserved tumor staging terms.",
            [
                "Doctors reviewed every generated summary before storage.",
                "The trial covered 320 anonymized reports.",
            ],
        ),
    },
    {
        "sample_id": "cm_021",
        "language_type": "mixed",
        "question": "苏州 factory QA benchmark 中 defect image captioning 用的是哪个系统？",
        "answers": ["VisionCN-Cap"],
        "keywords": ["苏州", "factory QA", "defect image captioning", "system"],
        "context": _build_context(
            [
                "Tianjin factory demo used a robot path planner.",
                "重庆团队展示了设备知识图谱。",
                "南京项目只处理质检单据OCR。",
            ],
            "In the 苏州 factory QA benchmark, VisionCN-Cap performed defect image captioning and produced Chinese inspection notes.",
            [
                "The dataset included scratches, stains, and missing screws.",
                "A human inspector checked 15% of outputs.",
            ],
        ),
    },
    {
        "sample_id": "cm_022",
        "language_type": "mixed",
        "question": "What was the bilingual search engine used in the Shanghai finance archive?",
        "answers": ["FinSearch-双语"],
        "keywords": ["Shanghai", "finance archive", "bilingual search", "engine"],
        "context": _build_context(
            [
                "Beijing bank tested loan document classification.",
                "深圳风控项目识别异常交易。",
                "Hangzhou demo focused on quarterly report tables.",
            ],
            "The Shanghai finance archive used FinSearch-双语 as the bilingual search engine for Chinese filings and English analyst notes.",
            [
                "The archive covered 2019 to 2024 reports.",
                "Analysts could filter by company, sector, and risk type.",
            ],
        ),
    },
    {
        "sample_id": "cm_023",
        "language_type": "mixed",
        "question": "Which assistant mapped English API errors to Chinese tickets in Chengdu?",
        "answers": ["TicketMap-成都"],
        "keywords": ["English API errors", "Chinese tickets", "Chengdu", "assistant"],
        "context": _build_context(
            [
                "Nanjing platform summarized user manuals.",
                "广州团队处理公交投诉分类。",
                "Shanghai system grouped server metrics by region.",
            ],
            "In Chengdu, TicketMap-成都 mapped English API errors to Chinese tickets and reduced duplicate bug reports.",
            [
                "The assistant kept stack traces unchanged for engineers.",
                "运营人员每周抽查50条映射结果。",
            ],
        ),
    },
    {
        "sample_id": "cm_024",
        "language_type": "mixed",
        "question": "杭州 education demo 中生成 bilingual quiz 的工具叫什么？",
        "answers": ["QuizBridge-HZ"],
        "keywords": ["杭州", "education demo", "bilingual quiz", "工具"],
        "context": _build_context(
            [
                "Wuhan school tested essay scoring.",
                "北京平台整理数学错题。",
                "Shenzhen app handled spoken English practice.",
            ],
            "At the 杭州 education demo, QuizBridge-HZ generated bilingual quiz items from Chinese textbook chapters and English glossaries.",
            [
                "Teachers could edit every generated question.",
                "The first trial used 48 history lessons.",
            ],
        ),
    },
    {
        "sample_id": "cm_025",
        "language_type": "mixed",
        "question": "What model aligned Chinese legal terms with English contract clauses?",
        "answers": ["LawAlign-CNEN"],
        "keywords": ["Chinese legal terms", "English contract clauses", "model"],
        "context": _build_context(
            [
                "Shanghai legal demo extracted invoice fields.",
                "广州项目分类劳动争议案件。",
                "Beijing team compared retrieval over court opinions.",
            ],
            "LawAlign-CNEN aligned Chinese legal terms with English contract clauses and highlighted mismatched liability phrases.",
            [
                "The evaluation set had 600 paired clauses.",
                "律师保留最终解释权。",
            ],
        ),
    },
    {
        "sample_id": "cm_026",
        "language_type": "mixed",
        "question": "青岛 ocean lab 用哪个 tool 预测 red tide alerts？",
        "answers": ["RedTide-Mix"],
        "keywords": ["青岛", "ocean lab", "red tide alerts", "tool"],
        "context": _build_context(
            [
                "Dalian group clustered ship trajectories.",
                "厦门项目分析游客密度。",
                "Ningbo dashboard tracked cold-chain containers.",
            ],
            "The 青岛 ocean lab used RedTide-Mix to predict red tide alerts from Chinese sensor logs and English weather reports.",
            [
                "Risk maps were refreshed every six hours.",
                "研究员记录了水温、盐度和叶绿素指标。",
            ],
        ),
    },
    {
        "sample_id": "cm_027",
        "language_type": "mixed",
        "question": "Which module checked validity dates in Changsha policy QA?",
        "answers": ["PolicyDate-星政"],
        "keywords": ["Changsha", "policy QA", "validity dates", "module"],
        "context": _build_context(
            [
                "Fuzhou hotline project merged duplicate requests.",
                "武汉政务系统支持地图服务搜索。",
                "Guangzhou platform recommended application materials.",
            ],
            "In Changsha policy QA, PolicyDate-星政 checked validity dates and flagged expired Chinese policy documents.",
            [
                "The module synchronized regulation updates every night.",
                "人工审核覆盖高频事项。",
            ],
        ),
    },
    {
        "sample_id": "cm_028",
        "language_type": "mixed",
        "question": "西安 safety platform 中 gas leak detection 使用了哪个模型？",
        "answers": ["GasSentinel-XA"],
        "keywords": ["西安", "safety platform", "gas leak detection", "模型"],
        "context": _build_context(
            [
                "Zhengzhou system warned about rainwater pooling.",
                "成都平台识别电梯困人事件。",
                "Chongqing dashboard analyzed bridge vibration.",
            ],
            "The 西安 safety platform used GasSentinel-XA for gas leak detection and combined Chinese alarm records with pressure readings.",
            [
                "Operators received alerts within five minutes.",
                "社区巡检记录将在二期接入。",
            ],
        ),
    },
    {
        "sample_id": "cm_029",
        "language_type": "mixed",
        "question": "天津 port scheduling 中 berth allocation 用的 algorithm 是什么？",
        "answers": ["HarborPlan-TJ"],
        "keywords": ["天津", "port scheduling", "berth allocation", "algorithm"],
        "context": _build_context(
            [
                "Qingdao port tracked cold-chain temperature.",
                "上海港区讨论无人集卡路线。",
                "Ningbo customs demo recognized shipping documents.",
            ],
            "In 天津 port scheduling, HarborPlan-TJ handled berth allocation and reduced average waiting time by 9.8%.",
            [
                "The algorithm used tide windows and vessel length.",
                "调度员仍然需要确认最终方案。",
            ],
        ),
    },
    {
        "sample_id": "cm_030",
        "language_type": "mixed",
        "question": "厦门 finance pilot 中 abnormal transfer review 的系统名称是什么？",
        "answers": ["RiskLens-XM"],
        "keywords": ["厦门", "finance pilot", "abnormal transfer review", "系统"],
        "context": _build_context(
            [
                "Shanghai bank summarized customer service chats.",
                "北京团队核验贷款申请材料。",
                "Shenzhen platform updated anti-money-laundering rules.",
            ],
            "In the 厦门 finance pilot, RiskLens-XM supported abnormal transfer review and escalated transactions above 300% of history average.",
            [
                "The pilot only produced risk hints and did not freeze accounts.",
                "审计日志保留90天。",
            ],
        ),
    },
]


def normalize_for_match(text: Any) -> str:
    text = "" if text is None else str(text)
    lowered = text.lower()
    return "".join(ch for ch in lowered if not ch.isspace())


def contains_any_answer(text: str, answers: list[str]) -> bool:
    normalized_text = normalize_for_match(text)
    return any(normalize_for_match(ans) in normalized_text for ans in answers)


def calculate_keyword_coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    normalized_text = normalize_for_match(text)
    hits = sum(
        1 for keyword in keywords if normalize_for_match(keyword) in normalized_text
    )
    return hits / len(keywords)


def run_compression_eval() -> pd.DataFrame:
    compressor = ContextCompressor()
    rows: list[dict[str, Any]] = []
    for sample in CHINESE_MIXED_DATASET:
        for method in METHODS:
            result = compressor.compress_context(
                context=sample["context"],
                question=sample["question"],
                method=method,
                token_budget=TOKEN_BUDGET,
            )
            compressed = result["compressed_context"]
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "language_type": sample["language_type"],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "question": sample["question"],
                    "answers": json.dumps(sample["answers"], ensure_ascii=False),
                    "keywords": json.dumps(sample["keywords"], ensure_ascii=False),
                    "contains_answer": int(
                        contains_any_answer(compressed, sample["answers"])
                    ),
                    "keyword_coverage": calculate_keyword_coverage(
                        compressed, sample["keywords"]
                    ),
                    "original_tokens": result["original_tokens"],
                    "compressed_tokens": result["compressed_tokens"],
                    "token_saving_ratio": result["token_saving_ratio"],
                    "compression_time": result["compression_time"],
                    "segment_count": len(split_into_segments(sample["context"])),
                    "compressed_context": compressed,
                }
            )
    return pd.DataFrame(rows)


def generate_nonempty_answer_with_retries(
    prompt: str,
    max_tokens: int = 96,
    attempts: int = 3,
    generator=generate_answer,
    sleep_fn=time.sleep,
) -> dict[str, Any]:
    last_error = ""
    last_result: dict[str, Any] = {
        "answer": "",
        "answer_time": 0.0,
        "usage": None,
        "retries": 0,
        "error_message": "",
    }

    for attempt in range(attempts):
        try:
            result = generator(prompt=prompt, max_tokens=max_tokens)
            answer = str(result.get("answer") or "").strip()
            last_result = {
                "answer": answer,
                "answer_time": float(result.get("answer_time") or 0.0),
                "usage": result.get("usage"),
                "retries": attempt,
                "error_message": "",
            }
            if answer:
                return last_result
            last_error = "empty_prediction"
        except Exception as exc:  # pragma: no cover - network-dependent path
            last_error = str(exc)
            last_result = {
                "answer": "",
                "answer_time": 0.0,
                "usage": None,
                "retries": attempt,
                "error_message": last_error,
            }
        if attempt < attempts - 1:
            sleep_fn(1 + attempt)

    last_result["error_message"] = last_error
    return last_result


def summarize_compression(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["method", "method_label", "language_type"])
        .agg(
            rows=("sample_id", "count"),
            contains_answer_rate=("contains_answer", "mean"),
            keyword_coverage=("keyword_coverage", "mean"),
            avg_compressed_tokens=("compressed_tokens", "mean"),
            avg_token_saving=("token_saving_ratio", "mean"),
            avg_compression_time=("compression_time", "mean"),
        )
        .reset_index()
    )
    overall = (
        df.groupby(["method", "method_label"])
        .agg(
            rows=("sample_id", "count"),
            contains_answer_rate=("contains_answer", "mean"),
            keyword_coverage=("keyword_coverage", "mean"),
            avg_compressed_tokens=("compressed_tokens", "mean"),
            avg_token_saving=("token_saving_ratio", "mean"),
            avg_compression_time=("compression_time", "mean"),
        )
        .reset_index()
    )
    overall.insert(2, "language_type", "overall")
    combined = pd.concat([overall, summary], ignore_index=True)
    combined["method_order"] = combined["method"].map(METHOD_RANK)
    combined = combined.sort_values(["language_type", "method_order"]).reset_index(
        drop=True
    )
    return combined.drop(columns=["method_order"])


def run_api_eval(compression_df: pd.DataFrame) -> pd.DataFrame:
    load_project_env()
    if os.getenv("QKP_CHINESE_SKIP_API", "").strip() == "1":
        existing = _read_existing_api_results()
        if not existing.empty:
            return existing
        return pd.DataFrame(columns=API_RESULT_COLUMNS)

    api_sample_ids = {
        sample["sample_id"] for sample in CHINESE_MIXED_DATASET[:API_SAMPLE_COUNT]
    }
    api_rows: list[dict[str, Any]] = []
    selected = compression_df[
        compression_df["sample_id"].isin(api_sample_ids)
        & compression_df["method"].isin(API_METHODS)
    ]

    for _, row in selected.iterrows():
        answers = json.loads(row["answers"])
        prompt = build_qa_prompt(row["compressed_context"], row["question"])
        result = generate_nonempty_answer_with_retries(prompt=prompt)
        prediction = str(result.get("answer") or "").strip()
        error_message = str(result.get("error_message") or "")
        answer_time = float(result.get("answer_time") or 0.0)
        retries = int(result.get("retries") or 0)
        usage = result.get("usage") or {}
        total_tokens = usage.get("total_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        api_rows.append(
            {
                "sample_id": row["sample_id"],
                "language_type": row["language_type"],
                "method": row["method"],
                "method_label": row["method_label"],
                "question": row["question"],
                "answers": row["answers"],
                "prediction": prediction,
                "api_contains_answer": int(contains_any_answer(prediction, answers)),
                "empty_prediction": int(not bool(prediction)),
                "error_message": error_message if not prediction else "",
                "answer_time": answer_time,
                "api_total_tokens": total_tokens,
                "api_prompt_tokens": prompt_tokens,
                "api_completion_tokens": completion_tokens,
                "retries": retries,
            }
        )
    return pd.DataFrame(api_rows)


def summarize_api(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=API_SUMMARY_COLUMNS)
    out = df.copy()
    out["has_error"] = out["error_message"].fillna("").astype(str).str.len() > 0
    by_language = (
        out.groupby(["method", "method_label", "language_type"])
        .agg(
            rows=("sample_id", "count"),
            api_contains_answer=("api_contains_answer", "mean"),
            empty_predictions=("empty_prediction", "sum"),
            errors=("has_error", "sum"),
            avg_answer_time=("answer_time", "mean"),
            avg_api_total_tokens=("api_total_tokens", "mean"),
        )
        .reset_index()
    )
    overall = (
        out.groupby(["method", "method_label"])
        .agg(
            rows=("sample_id", "count"),
            api_contains_answer=("api_contains_answer", "mean"),
            empty_predictions=("empty_prediction", "sum"),
            errors=("has_error", "sum"),
            avg_answer_time=("answer_time", "mean"),
            avg_api_total_tokens=("api_total_tokens", "mean"),
        )
        .reset_index()
    )
    overall.insert(2, "language_type", "overall")
    return pd.concat([overall, by_language], ignore_index=True)


def _read_existing_api_results() -> pd.DataFrame:
    """Reuse previous API results when API execution is explicitly skipped."""
    if not API_RESULT_OUT.exists() or API_RESULT_OUT.stat().st_size == 0:
        return pd.DataFrame(columns=API_RESULT_COLUMNS)

    try:
        existing = pd.read_csv(API_RESULT_OUT)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=API_RESULT_COLUMNS)

    if existing.empty:
        return pd.DataFrame(columns=API_RESULT_COLUMNS)

    for column in API_RESULT_COLUMNS:
        if column not in existing.columns:
            existing[column] = None

    return existing[API_RESULT_COLUMNS]


def write_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET_OUT.open("w", encoding="utf-8") as f:
        for sample in CHINESE_MIXED_DATASET:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_case_study(compression_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    sample = CHINESE_MIXED_DATASET[0]
    segments = split_into_segments(sample["context"])
    keywords = extract_question_keywords(sample["question"])

    lines = [
        "# QKP-Reorder Chinese/Mixed Case Study",
        "",
        f"- jieba_available: `{is_jieba_available()}`",
        f"- sample_id: `{sample['sample_id']}`",
        f"- question: {sample['question']}",
        f"- gold_answers: `{', '.join(sample['answers'])}`",
        f"- extracted_keywords: `{', '.join(keywords)}`",
        "",
        "## Compression Summary",
        "",
        summary_df[summary_df["language_type"] == "overall"]
        .sort_values("contains_answer_rate", ascending=False)
        .to_markdown(index=False),
        "",
        "## Segment-level Analysis",
        "",
        "| segment | tokens | protection_bonus |",
        "| --- | --- | ---: |",
    ]
    for seg in segments:
        tokens = tokenize_for_bm25(seg)
        bonus = compute_protection_bonus(seg, keywords)
        lines.append(f"| {seg} | {', '.join(tokens)} | {bonus:.2f} |")

    lines.extend(
        [
            "",
            "## Compressed Contexts",
            "",
        ]
    )
    rows = compression_df[compression_df["sample_id"] == sample["sample_id"]]
    for _, row in rows.iterrows():
        lines.extend(
            [
                f"### {row['method_label']}",
                "",
                f"- contains_answer: `{row['contains_answer']}`",
                f"- keyword_coverage: `{row['keyword_coverage']:.3f}`",
                "",
                "```text",
                str(row["compressed_context"]),
                "```",
                "",
            ]
        )

    CASE_STUDY_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_log(
    compression_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    api_df: pd.DataFrame,
    api_summary_df: pd.DataFrame,
) -> None:
    qkp = summary_df[
        (summary_df["method"] == "ours_full")
        & (summary_df["language_type"] == "overall")
    ].iloc[0]
    longllm = summary_df[
        (summary_df["method"] == "longllmlingua")
        & (summary_df["language_type"] == "overall")
    ].iloc[0]
    lines = [
        "QKP-Reorder Chinese/mixed validation",
        f"dataset_rows={len(CHINESE_MIXED_DATASET)}",
        f"compression_rows={len(compression_df)}",
        f"jieba_available={is_jieba_available()}",
        (
            "longllmlingua_contains_answer="
            f"{longllm['contains_answer_rate']:.4f}, "
            f"longllmlingua_keyword_coverage={longllm['keyword_coverage']:.4f}, "
            f"longllmlingua_avg_compressed_tokens="
            f"{longllm['avg_compressed_tokens']:.4f}, "
            f"longllmlingua_avg_token_saving="
            f"{longllm['avg_token_saving']:.4f}, "
            f"longllmlingua_avg_compression_time="
            f"{longllm['avg_compression_time']:.4f}"
        ),
        (
            "qkp_contains_answer="
            f"{qkp['contains_answer_rate']:.4f}, "
            f"qkp_keyword_coverage={qkp['keyword_coverage']:.4f}, "
            f"qkp_avg_token_saving={qkp['avg_token_saving']:.4f}"
        ),
        f"api_rows={len(api_df)}",
    ]
    if not api_summary_df.empty:
        qkp_api = api_summary_df[
            (api_summary_df["method"] == "ours_full")
            & (api_summary_df["language_type"] == "overall")
        ]
        if not qkp_api.empty:
            row = qkp_api.iloc[0]
            lines.append(
                "qkp_api_contains_answer="
                f"{row['api_contains_answer']:.4f}, "
                f"empty={int(row['empty_predictions'])}, "
                f"errors={int(row['errors'])}"
            )
    LOG_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_doc(summary_df: pd.DataFrame, api_summary_df: pd.DataFrame) -> None:
    overall = summary_df[summary_df["language_type"] == "overall"].copy()
    zh = summary_df[summary_df["language_type"] == "zh"].copy()
    mixed = summary_df[summary_df["language_type"] == "mixed"].copy()

    lines = [
        "# 阶段9 QKP-Reorder 中文与中英混合验证执行文档",
        "",
        "## 目标",
        "",
        "补充一个小型中文/中英混合验证集，专门验证 QKP-Reorder 的中文分句、中文关键词抽取、数字/实体保护和中英混合片段召回能力。",
        "",
        "## 数据集设计",
        "",
        "- 30 条人工构造长文问答样本。",
        "- 15 条纯中文，15 条中英混合。",
        "- 答案均放在中后段，前置多个干扰片段，用来测试简单截断是否容易漏掉答案。",
        "- 前 10 条额外运行 DeepSeek API 端到端问答小样本。",
        "",
        "## 压缩层结果",
        "",
        overall.to_markdown(index=False),
        "",
        "### 纯中文分组",
        "",
        zh.to_markdown(index=False),
        "",
        "### 中英混合分组",
        "",
        mixed.to_markdown(index=False),
        "",
        "## DeepSeek API 小样本",
        "",
    ]
    if api_summary_df.empty:
        lines.append("本次未运行 API 小样本。")
    else:
        lines.append(api_summary_df.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## 产物",
            "",
            f"- `{RESULT_OUT.relative_to(PROJECT_ROOT)}`",
            f"- `{SUMMARY_OUT.relative_to(PROJECT_ROOT)}`",
            f"- `{API_RESULT_OUT.relative_to(PROJECT_ROOT)}`",
            f"- `{API_SUMMARY_OUT.relative_to(PROJECT_ROOT)}`",
            f"- `{CASE_STUDY_OUT.relative_to(PROJECT_ROOT)}`",
            f"- `{LOG_OUT.relative_to(PROJECT_ROOT)}`",
            "",
            "## 报告写法建议",
            "",
            "这组实验可以支撑“QKP-Reorder 具备中文/中英混合处理能力”的说法，但不应替代 NQ、Lost-in-the-Middle 和 LongBench 的主实验结论。更准确的表述是：中文能力来自工程扩展与小样本验证，主实验优势仍主要体现在低成本、可解释和后位证据保留。",
        ]
    )
    DOC_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    write_dataset()
    compression_df = run_compression_eval()
    summary_df = summarize_compression(compression_df)
    compression_df.to_csv(RESULT_OUT, index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    api_df = run_api_eval(compression_df)
    api_summary_df = summarize_api(api_df)
    api_df.to_csv(API_RESULT_OUT, index=False, encoding="utf-8-sig")
    api_summary_df.to_csv(API_SUMMARY_OUT, index=False, encoding="utf-8-sig")

    write_case_study(compression_df, summary_df)
    write_log(compression_df, summary_df, api_df, api_summary_df)
    write_doc(summary_df, api_summary_df)

    print(f"wrote {RESULT_OUT}")
    print(f"wrote {SUMMARY_OUT}")
    print(f"wrote {API_RESULT_OUT}")
    print(f"wrote {API_SUMMARY_OUT}")
    print(f"wrote {CASE_STUDY_OUT}")
    print(f"wrote {DOC_OUT}")


if __name__ == "__main__":
    main()

