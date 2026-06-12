"""llm_client.py — 大模型 API 接入层（任务 3.3）

支持多厂商统一接口，调用方只需设置环境变量即可切换大模型，
无需修改任何业务代码。

支持厂商
--------
- 智谱 GLM（zhipu）  —— 推荐，国内访问稳定，有免费 API 额度
  https://open.bigmodel.cn/
- OpenAI 兼容接口   —— 适用于 OpenAI / DeepSeek / 月之暗面 / 阿里通义等
  任何遵循 OpenAI Chat Completion 格式的服务均可接入

环境变量配置
------------
必须设置（二选一即可启动）：
  LLM_PROVIDER=zhipu              # 或 openai
  ZHIPU_API_KEY=your_key          # 智谱平台申请：https://open.bigmodel.cn/
  OPENAI_API_KEY=your_key         # OpenAI 或其他兼容服务的 Key

可选设置：
  LLM_MODEL=glm-4-flash           # 默认模型（各厂商有不同默认值）
  LLM_BASE_URL=https://...        # 自定义接口地址（用于代理或国内镜像）
  LLM_TIMEOUT=60                  # 请求超时秒数（默认 60）
  LLM_MAX_TOKENS=1024             # 最大生成 token 数（默认 1024）
  LLM_TEMPERATURE=0.7             # 生成温度（默认 0.7）
  LLM_EMBEDDING_MODEL=...         # Embedding 模型名（用于文本转向量）

推荐配置示例（.env 或系统环境变量）：
  # 使用智谱 GLM（免费额度充足，推荐结项演示）
  LLM_PROVIDER=zhipu
  ZHIPU_API_KEY=xxxxxxxx.xxxxxxxx
  LLM_MODEL=glm-4-flash
  LLM_EMBEDDING_MODEL=embedding-3

  # 或使用 DeepSeek（OpenAI 兼容，性价比高）
  LLM_PROVIDER=openai
  OPENAI_API_KEY=sk-xxxxxxxx
  LLM_BASE_URL=https://api.deepseek.com/v1
  LLM_MODEL=deepseek-chat
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 可选依赖懒加载
# ---------------------------------------------------------------------------

try:
    from zhipuai import ZhipuAI as _ZhipuAI  # type: ignore
    _ZHIPU_AVAILABLE = True
except ImportError:
    _ZhipuAI = None  # type: ignore
    _ZHIPU_AVAILABLE = False

try:
    from openai import OpenAI as _OpenAI  # type: ignore
    _OPENAI_AVAILABLE = True
except ImportError:
    _OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# 默认模型配置
# ---------------------------------------------------------------------------

_DEFAULT_MODELS: Dict[str, Dict[str, str]] = {
    "zhipu": {
        "chat": "glm-4-flash",           # 速度快、有免费额度
        "embedding": "embedding-3",
    },
    "openai": {
        "chat": "gpt-3.5-turbo",
        "embedding": "text-embedding-3-small",
    },
}


# ---------------------------------------------------------------------------
# LLMClient 核心类
# ---------------------------------------------------------------------------

class LLMClient:
    """统一的大模型调用客户端。

    通过环境变量 ``LLM_PROVIDER`` 选择底层 SDK，
    对外暴露统一的 ``chat()`` 和 ``embed()`` 接口。

    Parameters
    ----------
    provider : str, optional
        大模型厂商，``"zhipu"`` 或 ``"openai"``（兼容任何 OpenAI 格式的服务）。
        默认从环境变量 ``LLM_PROVIDER`` 读取，未设置则自动探测已安装的 SDK。
    api_key : str, optional
        API 密钥，默认自动从对应环境变量读取。
    model : str, optional
        Chat 模型名称，默认从环境变量 ``LLM_MODEL`` 读取。
    base_url : str, optional
        自定义接口地址（用于代理、国内镜像或其他兼容服务）。
    timeout : float, optional
        HTTP 请求超时秒数，默认 60。
    max_tokens : int, optional
        最大生成 token 数，默认 1024。
    temperature : float, optional
        生成温度，默认 0.7。
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        self._provider = self._resolve_provider(provider)
        self._api_key = self._resolve_api_key(api_key)
        self._model = model or os.getenv("LLM_MODEL") or _DEFAULT_MODELS[self._provider]["chat"]
        self._embedding_model = (
            os.getenv("LLM_EMBEDDING_MODEL")
            or _DEFAULT_MODELS[self._provider]["embedding"]
        )
        self._base_url = base_url or os.getenv("LLM_BASE_URL") or None
        self._timeout = float(os.getenv("LLM_TIMEOUT", timeout))
        self._max_tokens = int(os.getenv("LLM_MAX_TOKENS", max_tokens))
        self._temperature = float(os.getenv("LLM_TEMPERATURE", temperature))

        # 惰性初始化，首次调用时创建
        self._client: Optional[Any] = None

        logger.info(
            "LLMClient 初始化: provider=%s, model=%s, base_url=%s",
            self._provider, self._model, self._base_url or "(default)",
        )

    # ------------------------------------------------------------------
    # 公开属性
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def is_available(self) -> bool:
        """检查依赖 SDK 是否已安装。"""
        if self._provider == "zhipu":
            return _ZHIPU_AVAILABLE
        return _OPENAI_AVAILABLE

    # ------------------------------------------------------------------
    # 核心接口：Chat Completion
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """调用 Chat Completion API，返回完整回复文本（非流式）。

        Parameters
        ----------
        messages:
            OpenAI 格式的消息列表，例如::

                [
                    {"role": "system", "content": "你是..."},
                    {"role": "user",   "content": "这些细胞是什么类型？"},
                ]
        model:
            覆盖默认模型名称。
        max_tokens:
            覆盖默认最大生成 token 数。
        temperature:
            覆盖默认生成温度。

        Returns
        -------
        str
            模型生成的回复文本。

        Raises
        ------
        RuntimeError
            SDK 未安装或 API 调用失败时抛出。
        """
        if not self.is_available:
            raise RuntimeError(
                f"SDK '{self._provider}' 未安装，请执行: "
                + ("pip install zhipuai" if self._provider == "zhipu" else "pip install openai")
            )

        use_model = model or self._model
        use_max_tokens = max_tokens if max_tokens is not None else self._max_tokens
        use_temperature = temperature if temperature is not None else self._temperature

        client = self._get_client()

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=use_model,
                messages=messages,
                max_tokens=use_max_tokens,
                temperature=use_temperature,
            )

            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            text = response.choices[0].message.content or ""
            logger.debug(
                "LLM chat 完成: provider=%s, model=%s, elapsed=%.0fms, tokens=%s",
                self._provider, use_model, elapsed,
                getattr(response, "usage", None),
            )
            return text.strip()

        except Exception as exc:
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            logger.error("LLM chat 失败 (%.0fms): %s", elapsed, exc)
            raise RuntimeError(f"大模型调用失败：{exc}") from exc

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """调用 Chat Completion API 流式版本，以生成器方式逐块 yield 文本片段。

        前端使用 SSE（Server-Sent Events）接收时，逐字显示更流畅。

        Parameters
        ----------
        messages:
            OpenAI 格式的消息列表。
        model / max_tokens / temperature:
            同 ``chat()``，可覆盖默认值。

        Yields
        ------
        str
            每次 yield 一个文本片段（delta），空片段会被跳过。

        Raises
        ------
        RuntimeError
            SDK 未安装或流式调用失败时抛出。

        Usage
        -----
            for chunk in client.stream_chat(messages):
                print(chunk, end="", flush=True)
        """
        if not self.is_available:
            raise RuntimeError(
                f"SDK '{self._provider}' 未安装，请执行: "
                + ("pip install zhipuai" if self._provider == "zhipu" else "pip install openai")
            )

        use_model = model or self._model
        use_max_tokens = max_tokens if max_tokens is not None else self._max_tokens
        use_temperature = temperature if temperature is not None else self._temperature

        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=use_model,
                messages=messages,
                max_tokens=use_max_tokens,
                temperature=use_temperature,
                stream=True,
            )
            for chunk in response:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
        except Exception as exc:
            logger.error("LLM stream_chat 失败: %s", exc)
            raise RuntimeError(f"大模型流式调用失败：{exc}") from exc

    # ------------------------------------------------------------------
    # 核心接口：文本 Embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """将文本转为向量（用于将用户问题转化为查询向量）。

        Parameters
        ----------
        text:
            待编码的文本字符串。
        model:
            覆盖默认 Embedding 模型名称。

        Returns
        -------
        List[float]
            向量列表（维度取决于模型）。

        Raises
        ------
        RuntimeError
            SDK 未安装或调用失败时抛出。
        """
        if not self.is_available:
            raise RuntimeError(
                f"SDK '{self._provider}' 未安装，"
                + ("请执行: pip install zhipuai" if self._provider == "zhipu" else "请执行: pip install openai")
            )

        use_model = model or self._embedding_model
        client = self._get_client()

        try:
            if self._provider == "zhipu":
                resp = client.embeddings.create(model=use_model, input=text)
                return resp.data[0].embedding
            else:
                resp = client.embeddings.create(model=use_model, input=text)
                return resp.data[0].embedding
        except Exception as exc:
            logger.error("LLM embed 失败: %s", exc)
            raise RuntimeError(f"Embedding 调用失败：{exc}") from exc

    # ------------------------------------------------------------------
    # 连通性测试
    # ------------------------------------------------------------------

    def ping(self) -> Dict[str, Any]:
        """发送一条极短的测试消息，验证 API Key 和网络连通性。

        Returns
        -------
        dict
            包含 ``ok``、``provider``、``model``、``elapsed_ms`` 等字段。
        """
        t0 = time.perf_counter()
        try:
            reply = self.chat(
                messages=[{"role": "user", "content": "Hi, reply with one word: OK"}],
                max_tokens=10,
                temperature=0.0,
            )
            return {
                "ok": True,
                "provider": self._provider,
                "model": self._model,
                "reply": reply,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider": self._provider,
                "model": self._model,
                "error": str(exc),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            }

    def get_info(self) -> Dict[str, Any]:
        """返回客户端配置摘要（不含 API Key 明文）。"""
        key = self._api_key or ""
        masked = key[:6] + "****" + key[-4:] if len(key) > 10 else "****"
        return {
            "provider": self._provider,
            "model": self._model,
            "embedding_model": self._embedding_model,
            "base_url": self._base_url or "(default)",
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "sdk_available": self.is_available,
            "api_key_set": bool(key),
            "api_key_preview": masked,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """惰性初始化底层 SDK 客户端。"""
        if self._client is not None:
            return self._client

        if self._provider == "zhipu":
            if not _ZHIPU_AVAILABLE or _ZhipuAI is None:
                raise RuntimeError("zhipuai 未安装，请执行: pip install zhipuai")
            self._client = _ZhipuAI(api_key=self._api_key)
        else:
            if not _OPENAI_AVAILABLE or _OpenAI is None:
                raise RuntimeError("openai 未安装，请执行: pip install openai")
            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = _OpenAI(**kwargs)

        return self._client

    def _resolve_provider(self, provider: Optional[str]) -> str:
        """决定使用哪个 LLM 厂商。"""
        if provider:
            p = provider.strip().lower()
            if p not in ("zhipu", "openai"):
                raise ValueError(f"不支持的 provider: {provider}，请选择 zhipu 或 openai")
            return p

        env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if env_provider in ("zhipu", "openai"):
            return env_provider

        # 自动探测：优先选择已安装的 SDK
        if _ZHIPU_AVAILABLE and os.getenv("ZHIPU_API_KEY"):
            return "zhipu"
        if _OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            return "openai"
        if _ZHIPU_AVAILABLE:
            return "zhipu"
        if _OPENAI_AVAILABLE:
            return "openai"

        # 两个都没装，默认设 zhipu（等到实际调用时报错）
        return "zhipu"

    def _resolve_api_key(self, api_key: Optional[str]) -> Optional[str]:
        """从参数或环境变量获取 API Key。"""
        if api_key:
            return api_key
        if self._provider == "zhipu":
            return os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY")
        return os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# 全局单例（供 Flask app 复用同一客户端连接）
# ---------------------------------------------------------------------------

_LLM_CLIENT: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取全局 LLMClient 单例，首次调用时按环境变量初始化。"""
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = LLMClient()
    return _LLM_CLIENT


def reset_llm_client() -> None:
    """重置全局单例（用于测试或运行时更换 API Key）。"""
    global _LLM_CLIENT
    _LLM_CLIENT = None
