"""rag_engine.py — RAG 完整流程引擎（任务 3.3）

本模块将向量数据库、Prompt 工程、LLM 调用三个环节串联为完整的
"问答闭环"，并额外管理对话历史，支持多轮连续对话。

完整 RAG 流程
-------------
用户问题（自然语言）
      │
      ▼  ① 查询向量化
      │  将用户问题转为向量：
      │  优先用 LLM Embedding API；若未配置则用
      │  数据集内某细胞向量作为 fallback（由调用方传入）
      │
      ▼  ② 向量检索
      │  CellVectorStore.query_similar() → Top-K 相似细胞
      │
      ▼  ③ Prompt 组装
      │  PromptBuilder.build_messages() → messages 列表
      │  格式：[system(角色+上下文), user(问题)]
      │
      ▼  ④ LLM 生成
      │  LLMClient.chat() → 大模型回复文本
      │
      ▼  ⑤ 返回结果
      {
        "answer": "...",          # 大模型回答
        "retrieved_cells": [...], # 检索到的细胞（供前端展示）
        "elapsed_ms": 1234,       # 总耗时
        "context_used": "...",    # 喂给大模型的上下文（调试用）
      }

多轮对话
--------
RAGEngine 内部维护一个会话历史（session_id → 消息列表），
前端每次请求带上 session_id 即可实现连续对话。
会话历史仅保留最近 N 轮，防止 Token 超限。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.llm_client import LLMClient, get_llm_client
from backend.prompt_builder import PromptBuilder, PromptTemplate
from backend.vector_store import CellVectorStore

logger = logging.getLogger(__name__)

# 最多保留的历史轮次（每轮 = 1 条 user + 1 条 assistant）
MAX_HISTORY_ROUNDS = 5

# 默认检索数量
DEFAULT_N_RESULTS = 5

# 会话最长闲置时间（秒），超过此时间未使用的会话将被自动清理
SESSION_TTL_SECONDS = 3600  # 1 小时


# ---------------------------------------------------------------------------
# 对话历史管理
# ---------------------------------------------------------------------------

class ChatHistory:
    """轻量级内存对话历史，用于多轮 RAG 对话。

    Parameters
    ----------
    max_rounds:
        最多保留的历史轮次，超出后自动丢弃最旧的一轮。
    ttl_seconds:
        会话的最长闲置时间（秒）。超过此时间未使用的会话在下次
        ``append`` 或 ``get`` 时会被自动清理，防止内存无限增长。
    """

    def __init__(
        self,
        max_rounds: int = MAX_HISTORY_ROUNDS,
        ttl_seconds: float = SESSION_TTL_SECONDS,
    ) -> None:
        self._max_rounds = max_rounds
        self._ttl = ttl_seconds
        # {session_id: [(user_msg, assistant_msg), ...]}
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        # {session_id: last_active_timestamp}
        self._last_active: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> List[Dict[str, str]]:
        """返回会话历史（仅含 user/assistant 消息，不含 system）。"""
        self._evict_expired()
        if session_id not in self._sessions:
            return []
        self._last_active[session_id] = time.monotonic()
        return list(self._sessions[session_id])

    def append(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """追加一轮对话，超出 max_rounds 时自动裁剪，并刷新活跃时间戳。"""
        self._evict_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        # 每轮存为两条独立消息
        self._sessions[session_id].append({"role": "user",      "content": user_msg})
        self._sessions[session_id].append({"role": "assistant", "content": assistant_msg})
        # 裁剪：每轮 = 2 条消息
        max_msgs = self._max_rounds * 2
        if len(self._sessions[session_id]) > max_msgs:
            self._sessions[session_id] = self._sessions[session_id][-max_msgs:]
        self._last_active[session_id] = time.monotonic()

    def clear(self, session_id: str) -> None:
        """清空指定会话的历史。"""
        self._sessions.pop(session_id, None)
        self._last_active.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    # ------------------------------------------------------------------
    # 内部清理
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        """清除所有超过 TTL 未活跃的会话，防止内存无限增长。"""
        if self._ttl <= 0:
            return
        now = time.monotonic()
        expired: List[str] = [
            sid
            for sid, last in self._last_active.items()
            if now - last > self._ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_active.pop(sid, None)
        if expired:
            logger.debug("ChatHistory: 清理了 %d 个过期会话", len(expired))


# 全局单例
_CHAT_HISTORY = ChatHistory()


# ---------------------------------------------------------------------------
# RAG 引擎核心
# ---------------------------------------------------------------------------

class RAGEngine:
    """单细胞 RAG 问答引擎。

    将向量数据库检索、Prompt 组装、LLM 调用串联为完整闭环。

    Parameters
    ----------
    vector_store:
        已初始化（populated）的 ``CellVectorStore`` 实例。
    llm_client:
        ``LLMClient`` 实例，负责调用大模型。
        ``None`` 时自动使用全局单例（由环境变量配置）。
    n_results:
        每次检索返回的细胞数量，默认 5。
    dataset_info:
        数据集背景说明（物种、组织等），注入 system prompt，
        例如 ``"人类肝脏单细胞图谱，物种：Homo sapiens"``。
    max_genes:
        Prompt 中显示的最大高表达基因数，默认 15。
    """

    def __init__(
        self,
        vector_store: CellVectorStore,
        llm_client: Optional[LLMClient] = None,
        n_results: int = DEFAULT_N_RESULTS,
        dataset_info: Optional[str] = None,
        max_genes: int = 15,
    ) -> None:
        self._store = vector_store
        self._llm = llm_client or get_llm_client()
        self._n_results = n_results
        self._dataset_info = dataset_info
        self._builder = PromptBuilder(
            template=PromptTemplate(max_genes_in_prompt=max_genes)
        )

    # ------------------------------------------------------------------
    # 公开属性
    # ------------------------------------------------------------------

    @property
    def dataset_info(self) -> Optional[str]:
        """注入 system prompt 的数据集背景说明。"""
        return self._dataset_info

    @property
    def prompt_builder(self) -> PromptBuilder:
        """内部 PromptBuilder 实例（供流式接口直接复用）。"""
        return self._builder

    # ------------------------------------------------------------------
    # 核心问答接口
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        query_vector: Optional[List[float]] = None,
        session_id: Optional[str] = None,
        n_results: Optional[int] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行完整 RAG 流程，返回结构化问答结果。

        Parameters
        ----------
        question:
            用户的自然语言问题。
        query_vector:
            查询向量（float 列表或 numpy 数组）。
        session_id:
            会话 ID，提供时启用多轮对话。
        n_results:
            本次检索数量。
        where_filter:
            ChromaDB 元数据过滤条件。
        keywords:
            可选关键词列表，用于向量检索不可用时的关键词回退检索。
        temperature:
            LLM 生成温度（0~1），None 使用默认值。
        max_tokens:
            最大生成 token 数，None 使用默认值。
        system_prompt:
            自定义系统 Prompt，None 使用模板默认值。
        """
        t_total = time.perf_counter()
        k = n_results or self._n_results

        # ── ① 向量化用户问题 ──────────────────────────────────
        query_vectorized = False
        if query_vector is None:
            query_vector = self._try_embed_question(question)
            query_vectorized = query_vector is not None

        # ── ② 向量检索（或关键词回退）──────────────────────
        t_retrieve = time.perf_counter()
        if query_vector is not None:
            retrieved = self._store.query_similar(
                query_vector=query_vector,
                n_results=k,
                where=where_filter,
            )
        elif keywords:
            logger.info("向量不可用，使用关键词检索: %s", keywords)
            retrieved = self._store.query_by_keywords(
                keywords=keywords, n_results=k,
            )
            if not retrieved:
                retrieved = self._store.query_by_metadata(where={}, limit=k)
        else:
            retrieved = self._store.query_by_metadata(where={}, limit=k)
        retrieve_ms = round((time.perf_counter() - t_retrieve) * 1000, 1)

        # ── ③ Prompt 组装 ────────────────────────────────────
        history = _CHAT_HISTORY.get(session_id) if session_id else None
        if history:
            messages = self._builder.build_messages_with_history(
                user_question=question,
                retrieved_cells=retrieved,
                history=history,
                extra_system_info=self._dataset_info,
                system_prompt=system_prompt,
            )
        else:
            messages = self._builder.build_messages(
                user_question=question,
                retrieved_cells=retrieved,
                extra_system_info=self._dataset_info,
                system_prompt=system_prompt,
            )

        # 提取上下文段落（system 消息中检索上下文之后的部分）
        context_used = self._builder.build_context(retrieved)

        # ── ④ LLM 生成 ───────────────────────────────────────
        t_llm = time.perf_counter()
        answer = self._llm.chat(
            messages=messages,
            temperature=temperature if temperature is not None else None,
            max_tokens=max_tokens if max_tokens is not None else None,
        )
        llm_ms = round((time.perf_counter() - t_llm) * 1000, 1)

        # ── ⑤ 保存对话历史 ───────────────────────────────────
        if session_id:
            _CHAT_HISTORY.append(session_id, question, answer)

        total_ms = round((time.perf_counter() - t_total) * 1000, 1)

        return {
            "answer": answer,
            "retrieved_cells": retrieved,
            "context_used": context_used,
            "query_vectorized": query_vectorized,
            "elapsed_ms": total_ms,
            "retrieve_ms": retrieve_ms,
            "llm_ms": llm_ms,
            "session_id": session_id,
            "model": self._llm.model,
            "n_retrieved": len(retrieved),
        }

    def clear_history(self, session_id: str) -> None:
        """清空指定会话的对话历史。"""
        _CHAT_HISTORY.clear(session_id)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """获取指定会话的对话历史（供前端展示）。"""
        return _CHAT_HISTORY.get(session_id)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _try_embed_question(self, question: str) -> Optional[List[float]]:
        """尝试用 LLM Embedding API 将问题转为向量，失败时返回 None。"""
        try:
            vec = self._llm.embed(question)
            logger.debug("问题 Embedding 成功，向量维度: %d", len(vec))
            return vec
        except Exception as exc:
            logger.warning("Embedding API 调用失败，将使用 query_vector 参数: %s", exc)
            return None


# ---------------------------------------------------------------------------
# 全局引擎注册表（供 Flask app 复用，避免每次请求重新初始化）
# ---------------------------------------------------------------------------

# {dataset_id: (RAGEngine, dataset_info, n_results)}
_ENGINE_REGISTRY: Dict[str, Tuple[RAGEngine, Optional[str], int]] = {}


def get_or_create_engine(
    vector_store: CellVectorStore,
    dataset_id: str,
    llm_client: Optional[LLMClient] = None,
    dataset_info: Optional[str] = None,
    n_results: int = DEFAULT_N_RESULTS,
) -> RAGEngine:
    """获取或创建指定数据集的 RAGEngine 单例。

    若 ``dataset_info`` 或 ``n_results`` 与缓存的值不同，
    会自动重建引擎以确保配置生效，不会复用过时的旧实例。

    Parameters
    ----------
    vector_store:
        已初始化的 ``CellVectorStore``。
    dataset_id:
        数据集 ID，用作引擎注册键。
    llm_client:
        ``LLMClient`` 实例，``None`` 时使用全局单例。
    dataset_info:
        注入 system prompt 的数据集背景说明。
    n_results:
        默认检索数量。
    """
    cached = _ENGINE_REGISTRY.get(dataset_id)
    if cached is not None:
        cached_engine, cached_info, cached_n = cached
        # 若关键配置未变化则直接复用
        if cached_info == dataset_info and cached_n == n_results:
            return cached_engine
        logger.debug(
            "RAGEngine 配置变更，重建引擎: dataset_id=%s", dataset_id
        )

    engine = RAGEngine(
        vector_store=vector_store,
        llm_client=llm_client,
        dataset_info=dataset_info,
        n_results=n_results,
    )
    _ENGINE_REGISTRY[dataset_id] = (engine, dataset_info, n_results)
    return engine


def clear_engine_registry() -> None:
    """清空引擎注册表（主要用于测试）。"""
    _ENGINE_REGISTRY.clear()
