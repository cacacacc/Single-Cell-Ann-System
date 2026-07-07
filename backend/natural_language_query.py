from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_LIMIT = 50
MAX_LIMIT = 500
VALUE_SCAN_LIMIT = 250

GENE_STOP_WORDS = {
    "AI", "API", "ANN", "CPU", "DNA", "GPU", "HTTP", "JSON", "LLM", "PCA",
    "RAG", "RNA", "SSE", "TOP", "UMAP", "URL", "VDB", "AND", "ARE", "THE",
    "FOR", "WITH", "FROM", "THIS", "THAT",
}

FIELD_ALIASES = {
    "cell_type": [
        "cell_type", "celltype", "cell ontology", "cell ontology class",
        "cell type", "type", "细胞类型", "细胞类群", "类型", "亚群",
    ],
    "tissue": ["tissue", "组织", "来源组织"],
    "organ": ["organ", "器官"],
    "condition": ["condition", "disease", "status", "group", "条件", "疾病", "状态", "分组"],
    "sample": ["sample", "样本"],
    "donor": ["donor", "patient", "个体", "供体", "患者"],
    "batch": ["batch", "批次"],
    "sex": ["sex", "gender", "性别"],
    "leiden": ["leiden", "cluster", "聚类", "簇"],
    "louvain": ["louvain"],
}


@dataclass
class NaturalQueryCondition:
    field: str
    value: str
    operator: str = "eq"
    source: str = "inferred"

    def to_dict(self) -> Dict[str, str]:
        return {
            "field": self.field,
            "value": self.value,
            "operator": self.operator,
            "source": self.source,
        }


@dataclass
class NaturalQueryPlan:
    question: str
    limit: int = DEFAULT_LIMIT
    conditions: List[NaturalQueryCondition] = field(default_factory=list)
    gene_keywords: List[str] = field(default_factory=list)
    text_keywords: List[str] = field(default_factory=list)
    seed_cell_id: Optional[str] = None
    mode: str = "metadata"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "limit": self.limit,
            "mode": self.mode,
            "seed_cell_id": self.seed_cell_id,
            "conditions": [item.to_dict() for item in self.conditions],
            "gene_keywords": self.gene_keywords,
            "text_keywords": self.text_keywords,
            "warnings": self.warnings,
        }


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return _clean_text(value).casefold()


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        text = _clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _parse_limit(question: str, fallback: int) -> int:
    text = question.casefold()
    patterns = [
        r"(?:top|前|最多|返回|找|查询|显示)\s*(\d{1,4})",
        r"(\d{1,4})\s*(?:个|条|cells?|细胞)",
        r"k\s*[=:]\s*(\d{1,4})",
        r"limit\s*[=:]\s*(\d{1,4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return max(1, min(int(match.group(1)), MAX_LIMIT))
    return max(1, min(int(fallback), MAX_LIMIT))


def _resolve_alias_to_field(alias: str, obs_columns: Sequence[str]) -> Optional[str]:
    alias_norm = alias.casefold().replace(" ", "").replace("_", "")
    columns_by_norm = {
        col.casefold().replace(" ", "").replace("_", ""): col for col in obs_columns
    }
    if alias_norm in columns_by_norm:
        return columns_by_norm[alias_norm]
    for canonical, aliases in FIELD_ALIASES.items():
        if alias_norm == canonical.casefold().replace("_", ""):
            aliases = [canonical] + aliases
        if any(alias_norm == item.casefold().replace(" ", "").replace("_", "") for item in aliases):
            if canonical in obs_columns:
                return canonical
            for col in obs_columns:
                col_norm = col.casefold().replace(" ", "").replace("_", "")
                if col_norm == canonical.casefold().replace("_", ""):
                    return col
    return None


def _preferred_columns(obs_columns: Sequence[str]) -> List[str]:
    preferred: List[str] = []
    for canonical in FIELD_ALIASES:
        field = _resolve_alias_to_field(canonical, obs_columns)
        if field and field not in preferred:
            preferred.append(field)
    return preferred


def _column_values(loader: Any, field: str, max_values: int = VALUE_SCAN_LIMIT) -> List[str]:
    try:
        series = loader.adata.obs[field]
    except Exception:
        return []
    values = []
    try:
        unique_values = series.dropna().astype(str).unique().tolist()
    except Exception:
        unique_values = []
    for value in unique_values:
        text = _clean_text(value)
        if text and text.lower() not in {"nan", "none", "null"}:
            values.append(text)
        if len(values) >= max_values:
            break
    return values


def _add_condition(
    conditions: List[NaturalQueryCondition],
    field: Optional[str],
    value: Optional[str],
    source: str,
    operator: str = "eq",
) -> None:
    if not field or value is None:
        return
    text = _clean_text(value).strip(" ,;，；。:：\"'")
    if not text:
        return
    key = (field.casefold(), text.casefold(), operator)
    existing = {(c.field.casefold(), c.value.casefold(), c.operator) for c in conditions}
    if key not in existing:
        conditions.append(NaturalQueryCondition(field=field, value=text, operator=operator, source=source))


def _extract_explicit_conditions(question: str, obs_columns: Sequence[str]) -> List[NaturalQueryCondition]:
    conditions: List[NaturalQueryCondition] = []
    field_terms = list(obs_columns)
    for canonical, aliases in FIELD_ALIASES.items():
        field_terms.extend([canonical] + aliases)
    field_terms = sorted(_unique_preserve_order(field_terms), key=len, reverse=True)

    for term in field_terms:
        field = _resolve_alias_to_field(term, obs_columns)
        if not field:
            continue
        escaped = re.escape(term)
        patterns = [
            rf"(?:{escaped})\s*(?:=|==|:|：|为|是|属于|等于)\s*([A-Za-z0-9_.+\-/ ]{{1,80}})",
            rf"(?:{escaped})\s*(?:包含|含有|contains?)\s*([A-Za-z0-9_.+\-/ ]{{1,80}})",
        ]
        for idx, pattern in enumerate(patterns):
            for match in re.finditer(pattern, question, flags=re.IGNORECASE):
                value = match.group(1)
                value = re.split(r"\s*(?:,|，|;|；|。|\band\b|\bor\b|且|并且|同时)\s*", value)[0]
                _add_condition(
                    conditions,
                    field,
                    value,
                    source="explicit",
                    operator="contains" if idx == 1 else "eq",
                )
    return conditions


def _infer_value_conditions(question: str, loader: Any, obs_columns: Sequence[str]) -> List[NaturalQueryCondition]:
    conditions: List[NaturalQueryCondition] = []
    q_norm = _norm(question)
    preferred = _preferred_columns(obs_columns)
    mentioned_fields = [
        field for field in obs_columns
        if field.casefold() in q_norm or any(alias.casefold() in q_norm for alias in FIELD_ALIASES.get(field, []))
    ]
    fields = _unique_preserve_order(mentioned_fields + preferred)

    for field in fields:
        values = sorted(_column_values(loader, field), key=len, reverse=True)
        if not values:
            continue
        for value in values:
            value_norm = value.casefold()
            if len(value_norm) < 2:
                continue
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(value_norm)}(?![A-Za-z0-9_])", q_norm):
                _add_condition(conditions, field, value, source="value_match")
                break
    return conditions


def _extract_genes(question: str) -> List[str]:
    candidates = re.findall(r"(?<![A-Za-z0-9_])([A-Z][A-Z0-9]{1,9})(?![A-Za-z0-9_])", question)
    genes = [item for item in candidates if item.upper() not in GENE_STOP_WORDS]
    return _unique_preserve_order(genes)


def _extract_text_keywords(question: str) -> List[str]:
    keywords = re.findall(r"[A-Za-z][A-Za-z0-9_.+\-/]{2,}", question)
    filtered = [kw for kw in keywords if kw.upper() not in GENE_STOP_WORDS]
    return _unique_preserve_order(filtered)[:10]


def _extract_seed_cell_id(question: str, loader: Any) -> Optional[str]:
    patterns = [
        r"(?:cell_id|cell id|细胞\s*ID|细胞)\s*[=:：]?\s*([A-Za-z0-9_.:\-]+)",
        r"(?:similar to|相似于|类似于|以)\s*([A-Za-z0-9_.:\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            try:
                loader.cell_index_from_id(candidate)
                return candidate
            except Exception:
                continue

    if not any(token in question.casefold() for token in ["cell", "细胞", "相似"]):
        return None
    tokens = re.findall(r"[A-Za-z0-9_.:\-]{3,}", question)
    for token in tokens:
        try:
            loader.cell_index_from_id(token)
            return token
        except Exception:
            continue
    return None


def parse_natural_cell_query(
    question: str,
    loader: Any,
    *,
    limit: int = DEFAULT_LIMIT,
    seed_cell_id: Optional[str] = None,
) -> NaturalQueryPlan:
    text = _clean_text(question)
    obs_columns = list(getattr(loader, "obs_columns", []) or [])
    parsed_limit = _parse_limit(text, limit)
    conditions = _extract_explicit_conditions(text, obs_columns)
    for condition in _infer_value_conditions(text, loader, obs_columns):
        _add_condition(conditions, condition.field, condition.value, condition.source, condition.operator)
    genes = _extract_genes(text)
    seed = seed_cell_id or _extract_seed_cell_id(text, loader)

    mode = "metadata"
    if seed:
        mode = "similarity"
    elif genes:
        mode = "keyword"
    elif not conditions:
        mode = "broad"

    return NaturalQueryPlan(
        question=text,
        limit=parsed_limit,
        conditions=conditions,
        gene_keywords=genes,
        text_keywords=_extract_text_keywords(text),
        seed_cell_id=seed,
        mode=mode,
    )


def _metadata_matches(cell_info: Dict[str, Any], conditions: Sequence[NaturalQueryCondition]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    for condition in conditions:
        actual = cell_info.get(condition.field)
        if actual is None:
            return False, reasons
        actual_text = _norm(actual)
        wanted = _norm(condition.value)
        if condition.operator == "contains":
            matched = wanted in actual_text
        else:
            matched = actual_text == wanted
        if not matched:
            return False, reasons
        reasons.append(f"{condition.field}={condition.value}")
    return True, reasons


def _result_from_cell(loader: Any, idx: int, rank: int, reasons: Optional[List[str]] = None) -> Dict[str, Any]:
    info = loader.get_cell_info(int(idx))
    return {
        "rank": rank,
        "cell_index": int(idx),
        "cell_id": info.get("cell_id"),
        "cell_type": info.get("cell_type", info.get("celltype", "unknown")),
        "metadata": info,
        "match_reasons": reasons or [],
    }


def _keyword_result_ids(store: Any, keywords: Sequence[str], n_results: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not keywords:
        return [], None
    if store is None or not getattr(store, "is_populated", lambda: False)():
        return [], "Vector DB is not initialized; gene/top-gene keyword matching was skipped."
    try:
        return store.query_by_keywords(list(keywords), n_results=n_results), None
    except Exception as exc:
        return [], f"Keyword query failed: {exc}"


def execute_natural_cell_query(
    plan: NaturalQueryPlan,
    loader: Any,
    *,
    store: Any = None,
    use_rep: str = "X_pca",
) -> Dict[str, Any]:
    limit = max(1, min(int(plan.limit or DEFAULT_LIMIT), MAX_LIMIT))
    warnings = list(plan.warnings)
    results: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    keyword_hits: List[Dict[str, Any]] = []
    keyword_terms = plan.gene_keywords or []
    if keyword_terms:
        keyword_hits, warning = _keyword_result_ids(store, keyword_terms, n_results=min(max(limit * 8, 25), MAX_LIMIT))
        if warning:
            warnings.append(warning)

    if plan.seed_cell_id and store is not None and getattr(store, "is_populated", lambda: False)():
        try:
            seed_idx = loader.cell_index_from_id(plan.seed_cell_id)
            actual_rep = use_rep if use_rep in getattr(loader, "available_reps", []) else "X_pca"
            if actual_rep not in getattr(loader, "available_reps", []):
                actual_rep = "X"
            query_vector = loader.get_vector(seed_idx, use_rep=actual_rep).tolist()
            candidates = store.query_similar(query_vector=query_vector, n_results=min(max(limit * 6, 25), MAX_LIMIT))
            for candidate in candidates:
                idx = int(candidate.get("cell_index", -1))
                if idx < 0:
                    continue
                cell_info = loader.get_cell_info(idx)
                matched, reasons = _metadata_matches(cell_info, plan.conditions)
                if not matched:
                    continue
                candidate = dict(candidate)
                candidate["rank"] = len(results) + 1
                candidate["metadata"] = cell_info
                candidate["match_reasons"] = ["similar_to=" + plan.seed_cell_id] + reasons
                cid = str(candidate.get("cell_id"))
                if cid not in seen:
                    seen.add(cid)
                    results.append(candidate)
                if len(results) >= limit:
                    break
        except Exception as exc:
            warnings.append(f"Similarity query failed: {exc}")
    elif plan.seed_cell_id:
        warnings.append("Vector DB is not initialized; similarity query was skipped.")

    if keyword_hits and len(results) < limit:
        for hit in keyword_hits:
            idx = int(hit.get("cell_index", -1))
            if idx < 0:
                continue
            cell_info = loader.get_cell_info(idx)
            matched, reasons = _metadata_matches(cell_info, plan.conditions)
            if not matched:
                continue
            hit = dict(hit)
            hit["rank"] = len(results) + 1
            hit["metadata"] = cell_info
            hit["match_reasons"] = [f"keyword={kw}" for kw in keyword_terms] + reasons
            cid = str(hit.get("cell_id"))
            if cid not in seen:
                seen.add(cid)
                results.append(hit)
            if len(results) >= limit:
                break

    if len(results) < limit and (plan.conditions or not keyword_terms):
        for idx in range(int(loader.n_cells)):
            cell_info = loader.get_cell_info(idx)
            matched, reasons = _metadata_matches(cell_info, plan.conditions)
            if not matched:
                continue
            result = _result_from_cell(loader, idx, len(results) + 1, reasons)
            cid = str(result.get("cell_id"))
            if cid in seen:
                continue
            seen.add(cid)
            results.append(result)
            if len(results) >= limit:
                break

    return {
        "plan": plan.to_dict() | {"warnings": warnings},
        "results": results,
        "count": len(results),
        "truncated": len(results) >= limit,
    }
