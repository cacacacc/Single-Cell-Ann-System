"""prompt_builder.py — 细胞特征 Prompt 工程模块（任务 3.2）

本模块负责将向量检索返回的单细胞数据（高表达基因、细胞类型注释、元数据）
转化为结构化的文本 Prompt，供大模型接口（任务 3.3）消费。

核心设计原则
------------
1. **信息密度**：优先放入细胞类型、高表达基因（最能描述细胞特征的信息）。
2. **令牌节约**：基因列表截断到前 K 个，可选字段按重要性排序，避免超出
   大模型的上下文窗口（通常 4096 / 8192 tokens）。
3. **结构清晰**：用编号、缩进和分隔线让大模型容易解析每条细胞记录。
4. **可扩展**：模板通过 PromptTemplate 数据类配置，调用方可替换任意字段。

数据流
------
vector_store.query_similar()
        │  returns List[Dict]  (cell_type, top_genes, metadata, distance, ...)
        ▼
PromptBuilder.build_context()
        │  returns str  (retrieved_context 段落)
        ▼
PromptBuilder.build_messages()
        │  returns List[Dict]  (OpenAI-compatible messages list)
        ▼
LLM API  (任务 3.3)

使用示例
--------
    from backend.prompt_builder import PromptBuilder

    builder = PromptBuilder()
    retrieved = store.query_similar(query_vec, n_results=5)
    messages = builder.build_messages(
        user_question="这批细胞最可能是什么类型？有什么功能？",
        retrieved_cells=retrieved,
    )
    # messages 可直接传给 OpenAI / ZhipuAI / Qwen 等 Chat API
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 可配置的 Prompt 模板
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    """Prompt 模板配置，调用方可自由替换任意字段。

    Attributes
    ----------
    system_prompt:
        系统消息，告知大模型角色与任务目标。
    context_header:
        检索上下文段落的标题行，插入在所有细胞条目之前。
    context_footer:
        检索上下文段落的结尾行（可空），用于收尾提示。
    cell_entry_template:
        单条细胞记录的格式字符串，支持以下占位符：
        {rank}           排名（1, 2, 3 ...）
        {cell_type}      细胞类型
        {similarity}     相似度百分比字符串（如 "92.3%"）
        {top_genes}      高表达基因列表（逗号分隔）
        {extra_fields}   其余元数据行（已格式化为多行缩进字符串）
    max_genes_in_prompt:
        写入 Prompt 的最大高表达基因数量（避免 Token 超限）。
    extra_meta_keys:
        需要写入 Prompt 的额外元数据字段列表（按顺序）。
        ``None`` 表示自动选取常见字段。
    """

    system_prompt: str = (
        "你是一位专业的单细胞转录组学（scRNA-seq）生物信息学分析专家。"
        "你的任务是根据系统后端已经实际执行的检索/查询结果（包含细胞类型注释、高表达基因列表"
        "及相关元数据），用中文准确、简洁地回答用户关于细胞功能、分型和生物学"
        "意义的问题。\n\n"
        "⚠️ 重要：你的整个回复必须严格使用以下 Markdown 格式，禁止输出纯文本！\n"
        "1. 必须以 ## 标题开头，对回答进行分段组织；\n"
        "2. 列举细胞类型、基因或功能点时，必须使用 - 无序列表，每条一行；\n"
        "3. 所有基因名（如 **ALB**）、细胞类型名（如 **Hepatocyte**）必须使用 **粗体**；\n"
        "4. 段落之间用空行分隔，禁止输出连续的长段落；\n"
        "5. 适当使用 > 引用块突出重要结论；\n"
        "6. 优先基于提供的检索数据作答，不要凭空编造；\n"
        "7. 若数据不足以支撑结论，明确说明不确定性；\n"
        "8. 不要声称自己是在模拟查询。系统提供给你的细胞条目就是后端真实查询结果；"
        "但你不能声称执行了系统没有提供的额外数据库查询或计算；\n"
        "9. 当用户要求列出查询到的细胞时，必须使用系统提供的真实 Cell ID，禁止只写"
        "「细胞 #1」「细胞 #2」这类占位符；\n"
        "10. 回答开头必须先用 Markdown 表格列出查询结果摘要，至少包含 Cell ID、细胞类型、"
        "命中条件和关键元数据，再进行功能解释。"
    )

    context_header: str = (
        "## 后端查询到的细胞数据（来自单细胞数据库）\n"
        "以下是系统后端根据本轮用户问题实际返回的 {n} 条细胞记录：\n"
    )

    context_footer: str = (
        "\n---\n"
        "请根据上述细胞数据回答用户的问题。若需要列出细胞，请优先列出真实 Cell ID，"
        "不要把排名编号当成细胞编号。"
    )

    cell_entry_template: str = (
        "[记录 #{rank}]\n"
        "  · Cell ID：{cell_id}\n"
        "  · Cell index：{cell_index}\n"
        "  · 细胞类型：{cell_type}{match_label}\n"
        "  · 高表达基因：{top_genes}\n"
        "{extra_fields}"
    )

    max_genes_in_prompt: int = 15

    extra_meta_keys: Optional[List[str]] = field(
        default_factory=lambda: [
            "tissue", "organ", "sample", "donor",
            "condition", "leiden", "louvain", "dataset",
        ]
    )


# 默认单例模板（可直接导入复用）
DEFAULT_TEMPLATE = PromptTemplate()


# ---------------------------------------------------------------------------
# 核心 Prompt 构建器
# ---------------------------------------------------------------------------

class PromptBuilder:
    """将向量检索结果组装为大模型可消费的 Prompt。

    Parameters
    ----------
    template:
        ``PromptTemplate`` 实例，控制格式与长度。默认使用 ``DEFAULT_TEMPLATE``。
    """

    def __init__(self, template: Optional[PromptTemplate] = None) -> None:
        self._tmpl = template or DEFAULT_TEMPLATE

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def build_context(
        self,
        retrieved_cells: List[Dict[str, Any]],
    ) -> str:
        """将检索结果列表转化为检索上下文文本段落。

        Parameters
        ----------
        retrieved_cells:
            ``CellVectorStore.query_similar()`` 的返回值列表，
            每个 dict 至少包含 ``cell_type``、``top_genes``、
            ``distance``、``metadata`` 字段。

        Returns
        -------
        str
            供插入 Prompt 的多行文本，包含标题、各细胞条目和结尾提示。
        """
        if not retrieved_cells:
            return "（未检索到相关细胞数据）"

        n = len(retrieved_cells)
        header = self._tmpl.context_header.format(n=n)
        entries = [
            self._format_cell_entry(cell, rank=i + 1)
            for i, cell in enumerate(retrieved_cells)
        ]
        footer = self._tmpl.context_footer

        return header + "\n".join(entries) + footer

    def build_messages(
        self,
        user_question: str,
        retrieved_cells: List[Dict[str, Any]],
        extra_system_info: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """构建 OpenAI-compatible messages 列表，直接传给 Chat Completion API。

        消息结构
        --------
        [
            {"role": "system",    "content": <系统 Prompt + 检索上下文>},
            {"role": "user",      "content": <用户原始问题>},
        ]

        将检索上下文放在 system 消息末尾（而非 user 消息），
        可减少大模型"角色混淆"的概率，同时充分利用系统消息的
        更高优先级指令权重。

        Parameters
        ----------
        user_question:
            用户的自然语言问题（原始文本）。
        retrieved_cells:
            ``CellVectorStore.query_similar()`` 返回值。
        extra_system_info:
            可选的数据集背景说明（物种、组织、实验条件等）。
        system_prompt:
            自定义系统 Prompt。若提供则替换模板默认值。
            可通过 ``get_preset_prompt()`` 获取预设角色 Prompt。

        Returns
        -------
        List[Dict[str, str]]
            符合 OpenAI / ZhipuAI / Qwen 等 Chat API 格式的消息列表。
        """
        context = self.build_context(retrieved_cells)

        system_parts = [system_prompt or self._tmpl.system_prompt]
        if extra_system_info:
            system_parts.append(f"\n\n【数据集背景信息】\n{extra_system_info}")
        # 在 system 消息中补充本次检索结果的细胞类型分布摘要
        type_summary = self._cell_type_summary(retrieved_cells)
        if type_summary:
            system_parts.append(f"\n\n【检索结果细胞类型分布】\n{type_summary}")
        system_parts.append(f"\n\n{context}")

        return [
            {"role": "system", "content": "\n".join(system_parts)},
            {"role": "user",   "content": user_question},
        ]

    def build_messages_with_history(
        self,
        user_question: str,
        retrieved_cells: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        extra_system_info: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """在多轮对话场景下构建带历史消息的 messages 列表。

        Parameters
        ----------
        user_question:
            当前轮次用户的问题。
        retrieved_cells:
            本轮检索到的细胞数据（每轮独立检索，上下文随之更新）。
        history:
            历史消息列表，格式与 messages 相同（不含 system 消息）。
            例如 ``[{"role": "user", ...}, {"role": "assistant", ...}, ...]``。
        extra_system_info:
            可选的额外系统说明。
        system_prompt:
            自定义系统 Prompt。若提供则替换模板默认值。

        Returns
        -------
        List[Dict[str, str]]
        """
        base = self.build_messages(
            user_question="",          # 占位，后面替换
            retrieved_cells=retrieved_cells,
            extra_system_info=extra_system_info,
            system_prompt=system_prompt,
        )
        system_msg = base[0]           # 取出 system 消息（已含细胞类型分布摘要）

        messages: List[Dict[str, str]] = [system_msg]
        if history:
            # 只保留 role in {user, assistant} 的历史
            messages.extend(
                m for m in history
                if m.get("role") in {"user", "assistant"}
            )
        messages.append({"role": "user", "content": user_question})
        return messages

    def format_single_cell(
        self,
        cell: Dict[str, Any],
        rank: int = 1,
    ) -> str:
        """格式化单条细胞数据为可读文本（调试 / 单独使用时）。

        Parameters
        ----------
        cell:
            单条检索结果 dict。
        rank:
            显示排名。

        Returns
        -------
        str
        """
        return self._format_cell_entry(cell, rank=rank)

    # ------------------------------------------------------------------
    # 内部格式化方法
    # ------------------------------------------------------------------

    def _format_cell_entry(self, cell: Dict[str, Any], rank: int) -> str:
        """将单条细胞检索结果格式化为模板字符串。"""
        metadata = cell.get("metadata") or {}
        cell_id = str(cell.get("cell_id") or metadata.get("cell_id") or "未知")
        cell_index = cell.get("cell_index", metadata.get("cell_index", "未知"))
        cell_type = str(cell.get("cell_type") or "未知细胞类型")
        match_label = self._format_match_label(cell)

        # 高表达基因处理
        raw_genes = cell.get("top_genes") or ""
        top_genes = self._format_top_genes(raw_genes)

        extra_fields = self._format_extra_fields(metadata)

        return self._tmpl.cell_entry_template.format(
            rank=rank,
            cell_id=cell_id,
            cell_index=cell_index,
            cell_type=cell_type,
            match_label=match_label,
            top_genes=top_genes,
            extra_fields=extra_fields,
        )

    def _format_match_label(self, cell: Dict[str, Any]) -> str:
        """Format similarity or deterministic match reasons for one cell."""
        if cell.get("distance") is not None:
            try:
                distance = float(cell.get("distance"))
                return f"（相似度 {self._distance_to_similarity_str(distance)}）"
            except (TypeError, ValueError):
                return "（相似度 N/A）"

        reasons = cell.get("match_reasons") or []
        if reasons:
            return "（命中：" + "；".join(str(item) for item in reasons[:4]) + "）"
        return "（条件匹配）"

    def _format_top_genes(self, raw_genes: str) -> str:
        """截断基因列表并格式化为可读字符串。

        输入示例：``"ALB,APOA1,CYP3A4,FABP1,HP,..."``
        输出示例：``"ALB、APOA1、CYP3A4、FABP1、HP ...（共 20 个）"``
        """
        if not raw_genes:
            return "（暂无基因数据）"

        genes = [g.strip() for g in raw_genes.split(",") if g.strip()]
        total = len(genes)

        max_k = self._tmpl.max_genes_in_prompt
        shown = genes[:max_k]
        gene_str = "、".join(shown)

        if total > max_k:
            return f"{gene_str} ...（共 {total} 个高表达基因）"
        return gene_str

    def _format_extra_fields(self, metadata: Dict[str, Any]) -> str:
        """将元数据中的可选字段格式化为缩进行。

        示例输出（每行以两个空格缩进）::

              · 组织来源：liver
              · 样本：donor_1
              · 分群（leiden）：2
        """
        keys_to_labels: Dict[str, str] = {
            "tissue":    "组织来源",
            "organ":     "器官",
            "sample":    "样本",
            "donor":     "供体",
            "condition": "实验条件",
            "leiden":    "分群（leiden）",
            "louvain":   "分群（louvain）",
            "dataset":   "数据集",
        }

        target_keys = self._tmpl.extra_meta_keys or list(keys_to_labels.keys())

        lines: List[str] = []
        for key in target_keys:
            if key not in metadata:
                continue
            val = metadata[key]
            if val is None or str(val).strip() in ("", "nan", "None"):
                continue
            label = keys_to_labels.get(key, key)
            lines.append(f"  · {label}：{val}")

        if not lines:
            return ""
        return "\n".join(lines) + "\n"

    def _cell_type_summary(self, retrieved_cells: List[Dict[str, Any]]) -> str:
        """统计检索结果中的细胞类型分布，生成简短摘要行。

        例如：Hepatocyte × 3、T cell × 1、Kupffer cell × 1

        Parameters
        ----------
        retrieved_cells:
            ``CellVectorStore.query_similar()`` 返回值。

        Returns
        -------
        str
            细胞类型分布的逗号分隔摘要；若只有一种类型则省略（避免冗余）。
        """
        if not retrieved_cells:
            return ""

        counts: Dict[str, int] = {}
        for cell in retrieved_cells:
            ct = str(cell.get("cell_type") or "未知").strip()
            counts[ct] = counts.get(ct, 0) + 1

        # 只有一种细胞类型时不额外输出（上下文条目本身已含类型）
        if len(counts) <= 1:
            return ""

        # 按数量降序排列（每行一条，便于 LLM 解析）
        parts = [
            f"- {ct}：{n} 个细胞"
            for ct, n in sorted(counts.items(), key=lambda x: -x[1])
        ]
        return "\n".join(parts)

    def _distance_to_similarity_str(self, distance: float) -> str:
        """将 ChromaDB 返回的距离值转为可读相似度百分比字符串。

        ChromaDB cosine 距离范围 [0, 2]，0 表示完全相同。
        转换公式：similarity = 1 - distance / 2，映射到 [0%, 100%]。
        """
        try:
            # cosine distance ∈ [0, 2] → similarity ∈ [0, 1]
            sim = max(0.0, min(1.0, 1.0 - distance / 2.0))
            return f"{sim * 100:.1f}%"
        except (TypeError, ValueError):
            return "N/A"


# ---------------------------------------------------------------------------
# 便捷工厂函数
# ---------------------------------------------------------------------------

def build_rag_messages(
    user_question: str,
    retrieved_cells: List[Dict[str, Any]],
    extra_system_info: Optional[str] = None,
    max_genes: int = 15,
) -> List[Dict[str, str]]:
    """模块级快捷函数：一行代码完成"检索结果 → messages"转化。

    Parameters
    ----------
    user_question:
        用户原始问题。
    retrieved_cells:
        ``CellVectorStore.query_similar()`` 返回值。
    extra_system_info:
        可选的数据集背景说明（物种、组织、实验条件等）。
    max_genes:
        Prompt 中显示的最大基因数量。

    Returns
    -------
    List[Dict[str, str]]
        可直接传给大模型 Chat API 的 messages 列表。

    示例
    ----
        from backend.prompt_builder import build_rag_messages

        messages = build_rag_messages(
            user_question="这些细胞主要执行哪些生物学功能？",
            retrieved_cells=store.query_similar(q_vec, n_results=5),
            extra_system_info="数据集：人类肝脏单细胞图谱，物种：Homo sapiens",
        )
    """
    tmpl = PromptTemplate(max_genes_in_prompt=max_genes)
    builder = PromptBuilder(template=tmpl)
    return builder.build_messages(
        user_question=user_question,
        retrieved_cells=retrieved_cells,
        extra_system_info=extra_system_info,
    )


# ---------------------------------------------------------------------------
# 系统角色预设（供前端 LLM Controls 选择）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PRESETS: Dict[str, Dict[str, str]] = {
    "bioinfo_expert": {
        "label": "生信分析专家",
        "icon": "dna",
        "description": "专业全面地分析单细胞数据，兼顾统计与生物学解读",
        "prompt": (
            "你是一位专业的单细胞转录组学（scRNA-seq）生物信息学分析专家。"
            "你的任务是根据系统后端已经实际执行的检索/查询结果（包含细胞类型注释、高表达基因列表"
            "及相关元数据），用中文准确、全面地回答用户关于细胞功能、分型和生物学"
            "意义的问题。\n\n"
            "⚠️ 重要：你的整个回复必须严格使用以下 Markdown 格式，禁止输出纯文本！\n"
            "1. 必须以 ## 标题开头，对回答进行分段组织；\n"
            "2. 列举细胞类型、基因或功能点时，必须使用 - 无序列表，每条一行；\n"
            "3. 所有基因名（如 **ALB**）、细胞类型名（如 **Hepatocyte**）必须使用 **粗体**；\n"
            "4. 段落之间用空行分隔，禁止输出连续的长段落；\n"
            "5. 适当使用 > 引用块突出重要结论；\n"
            "6. 优先基于提供的检索数据作答，不要凭空编造；\n"
            "7. 若数据不足以支撑结论，明确说明不确定性；\n"
            "8. 不要声称自己是在模拟查询。系统提供给你的细胞条目就是后端真实查询结果；"
            "但你不能声称执行了系统没有提供的额外数据库查询或计算；\n"
            "9. 当用户要求列出查询到的细胞时，必须使用系统提供的真实 Cell ID，禁止只写"
            "「细胞 #1」「细胞 #2」这类占位符；\n"
            "10. 回答开头必须先用 Markdown 表格列出查询结果摘要，至少包含 Cell ID、细胞类型、"
            "命中条件和关键元数据，再进行功能解释。"
        ),
    },
    "strict_analyst": {
        "label": "严谨的数据统计员",
        "icon": "chart-bar",
        "description": "只基于检索数据做客观统计，不做延伸推测",
        "prompt": (
            "你是一位严谨的单细胞数据统计分析员。你的唯一职责是基于系统后端已经实际执行的检索/查询结果"
            "进行客观的统计描述。\n\n"
            "⚠️ 重要：你的整个回复必须严格使用以下 Markdown 格式，禁止输出纯文本！\n"
            "1. 必须以 ## 标题开头（如 ## 细胞类型分布、## 基因表达统计、## 元数据摘要）；\n"
            "2. 所有陈述必须直接引用检索数据中的具体数值，不得使用模糊用语；\n"
            "3. 使用 - 无序列表呈现统计数据，每条数据一行；\n"
            "4. 关键数值使用 **粗体** 突出；\n"
            "5. 严禁做任何生物学推测、功能解读或延伸假设；\n"
            "6. 若某项数据缺失，直接写「数据缺失」而非猜测填补；\n"
            "7. 不要使用「可能」「或许」「推测」等不确定性词汇，只陈述确定的事实；\n"
            "8. 不要声称自己是在模拟查询。系统提供给你的细胞条目就是后端真实查询结果；\n"
            "9. 必须使用系统提供的真实 Cell ID，禁止只写「细胞 #1」「细胞 #2」这类占位符；\n"
            "10. 回答开头必须先用 Markdown 表格列出查询结果摘要，至少包含 Cell ID、细胞类型、"
            "命中条件和关键元数据。"
        ),
    },
    "science_communicator": {
        "label": "通俗科普助手",
        "icon": "book-open",
        "description": "用通俗易懂的语言向非专业读者解释细胞数据",
        "prompt": (
            "你是一位擅长单细胞组学知识科普的沟通专家。你的任务是用通俗易懂、"
            "生动形象的语言，将复杂的单细胞数据解释给非专业读者（如患者、学生、公众）。\n\n"
            "⚠️ 重要：你的整个回复必须严格使用以下 Markdown 格式，禁止输出纯文本！\n"
            "1. 必须以 ## 标题开头，每个段落聚焦一个主题；\n"
            "2. 用生活化的比喻解释专业概念（如将细胞比作工厂、将基因比作指令）；\n"
            "3. 列举要点时使用 - 无序列表，每条控制在 1-2 句话；\n"
            "4. 关键术语首次出现时使用 **粗体**，并用括号加简短解释；\n"
            "5. 避免堆砌专业缩写，必须使用时先展开全称；\n"
            "6. 回答末尾用 > 引用块加一句总结；\n"
            "7. 基于系统后端已经实际执行的检索/查询结果作答，不编造信息，不确定的地方坦诚说明；\n"
            "8. 不要声称自己是在模拟查询。系统提供给你的细胞条目就是后端真实查询结果；\n"
            "9. 当列出查询到的细胞时，必须使用系统提供的真实 Cell ID，禁止只写"
            "「细胞 #1」「细胞 #2」这类占位符。"
        ),
    },
}

# 默认角色 key
DEFAULT_PRESET_KEY = "bioinfo_expert"


def get_preset_prompt(preset_key: str) -> str:
    """获取指定角色预设的 system prompt 文本。

    Parameters
    ----------
    preset_key:
        预设键名，如 ``"bioinfo_expert"``、``"strict_analyst"``、
        ``"science_communicator"``。

    Returns
    -------
    str
        对应的 system prompt 文本；若 key 不存在则返回默认预设。
    """
    preset = SYSTEM_PROMPT_PRESETS.get(
        preset_key, SYSTEM_PROMPT_PRESETS[DEFAULT_PRESET_KEY]
    )
    return preset["prompt"]
