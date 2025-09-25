"""
OpenAI兼容适配器基类
为所有支持OpenAI接口的LLM提供商提供统一的基础实现
"""

import os
import time
from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManagerForLLMRun

# 导入统一日志系统
from tradingagents.utils.logging_init import setup_llm_logging

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger, get_logger_manager
logger = get_logger('agents')
logger = setup_llm_logging()

# 导入token跟踪器
try:
    from tradingagents.config.config_manager import token_tracker
    TOKEN_TRACKING_ENABLED = True
    logger.info("✅ Token跟踪功能已启用")
except ImportError:
    TOKEN_TRACKING_ENABLED = False
    logger.warning("⚠️ Token跟踪功能未启用")


class OpenAICompatibleBase(ChatOpenAI):
    """
    OpenAI兼容适配器基类
    为所有支持OpenAI接口的LLM提供商提供统一实现
    """
    
    def __init__(
        self,
        provider_name: str,
        model: str,
        api_key_env_var: str,
        base_url: str,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        初始化OpenAI兼容适配器
        
        Args:
            provider_name: 提供商名称 (如: "deepseek", "dashscope")
            model: 模型名称
            api_key_env_var: API密钥环境变量名
            base_url: API基础URL
            api_key: API密钥，如果不提供则从环境变量获取
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数
        """
        
        # 在父类初始化前先缓存元信息到私有属性（避免Pydantic字段限制）
        object.__setattr__(self, "_provider_name", provider_name)
        object.__setattr__(self, "_model_name_alias", model)
        
        # 获取API密钥
        if api_key is None:
            api_key = os.getenv(api_key_env_var)
            if not api_key:
                raise ValueError(
                    f"{provider_name} API密钥未找到。"
                    f"请设置{api_key_env_var}环境变量或传入api_key参数。"
                )
        
        # 设置OpenAI兼容参数
        # 注意：model参数会被Pydantic映射到model_name字段
        openai_kwargs = {
            "model": model,  # 这会被映射到model_name字段
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # 根据LangChain版本使用不同的参数名
        try:
            # 新版本LangChain
            openai_kwargs.update({
                "api_key": api_key,
                "base_url": base_url
            })
        except:
            # 旧版本LangChain
            openai_kwargs.update({
                "openai_api_key": api_key,
                "openai_api_base": base_url
            })
        
        # 初始化父类
        super().__init__(**openai_kwargs)

        # 再次确保元信息存在（有些实现会在super()中重置__dict__）
        object.__setattr__(self, "_provider_name", provider_name)
        object.__setattr__(self, "_model_name_alias", model)

        logger.info(f"✅ {provider_name} OpenAI兼容适配器初始化成功")
        logger.info(f"   模型: {model}")
        logger.info(f"   API Base: {base_url}")

    @property
    def provider_name(self) -> Optional[str]:
        return getattr(self, "_provider_name", None)

    # 移除model_name property定义，使用Pydantic字段
    # model_name字段由ChatOpenAI基类的Pydantic字段提供
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        生成聊天响应，并记录token使用量
        """
        
        # 记录开始时间
        start_time = time.time()
        
        # 调用父类生成方法
        result = super()._generate(messages, stop, run_manager, **kwargs)
        
        # 记录token使用
        self._track_token_usage(result, kwargs, start_time)
        
        return result

    def _track_token_usage(self, result: ChatResult, kwargs: Dict, start_time: float):
        """记录token使用量并输出日志"""
        if not TOKEN_TRACKING_ENABLED:
            return
        try:
            # 统计token信息
            usage = getattr(result, "usage_metadata", None)
            total_tokens = usage.get("total_tokens") if usage else None
            prompt_tokens = usage.get("input_tokens") if usage else None
            completion_tokens = usage.get("output_tokens") if usage else None

            elapsed = time.time() - start_time
            logger.info(
                f"📊 Token使用 - Provider: {getattr(self, 'provider_name', 'unknown')}, Model: {getattr(self, 'model_name', 'unknown')}, "
                f"总tokens: {total_tokens}, 提示: {prompt_tokens}, 补全: {completion_tokens}, 用时: {elapsed:.2f}s"
            )
        except Exception as e:
            logger.warning(f"⚠️ Token跟踪记录失败: {e}")


class ChatDeepSeekOpenAI(OpenAICompatibleBase):
    """DeepSeek OpenAI兼容适配器"""
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            provider_name="deepseek",
            model=model,
            api_key_env_var="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


class ChatKimiOpenAI(OpenAICompatibleBase):
    """Kimi OpenAI兼容适配器"""
    
    def __init__(
        self,
        model: str = "kimi-chat",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            provider_name="kimi",
            model=model,
            api_key_env_var="KIMI_API_KEY",
            base_url="https://api.moonshot.cn",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


class ChatDashScopeOpenAIUnified(OpenAICompatibleBase):
    """阿里百炼 DashScope OpenAI兼容适配器"""
    
    def __init__(
        self,
        model: str = "qwen-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            provider_name="dashscope",
            model=model,
            api_key_env_var="DASHSCOPE_API_KEY",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


class ChatQianfanOpenAI(OpenAICompatibleBase):
    """文心一言千帆平台 OpenAI兼容适配器"""
    
    def __init__(
        self,
        model: str = "ERNIE-Speed-8K",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        # 千帆需要同时使用ACCESS_KEY和SECRET_KEY进行认证
        # 为了兼容OpenAI格式，我们将ACCESS_KEY作为api_key，SECRET_KEY通过环境变量获取
        access_key = api_key or os.getenv('QIANFAN_ACCESS_KEY')
        secret_key = os.getenv('QIANFAN_SECRET_KEY')
        
        if not access_key or not secret_key:
            raise ValueError(
                "千帆模型需要设置QIANFAN_ACCESS_KEY和QIANFAN_SECRET_KEY环境变量"
            )
        
        super().__init__(
            provider_name="qianfan",
            model=model,
            api_key_env_var="QIANFAN_ACCESS_KEY",
            base_url="https://qianfan.baidubce.com/v2",
            api_key=access_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


class ChatCustomOpenAI(OpenAICompatibleBase):
    """自定义OpenAI端点适配器（代理/聚合平台）"""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        if base_url is None:
            base_url = os.getenv("CUSTOM_OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        super().__init__(
            provider_name="custom_openai",
            model=model,
            api_key_env_var="CUSTOM_OPENAI_API_KEY",
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


# 支持的OpenAI兼容模型配置
OPENAI_COMPATIBLE_PROVIDERS = {
    "deepseek": {
        "adapter_class": ChatDeepSeekOpenAI,
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "models": {
            "deepseek-chat": {"context_length": 32768, "supports_function_calling": True},
            "deepseek-coder": {"context_length": 16384, "supports_function_calling": True}
        }
    },
    "kimi": {
        "adapter_class": ChatKimiOpenAI,
        "base_url": "https://api.moonshot.cn",
        "api_key_env": "KIMI_API_KEY",
        "models": {
            "moonshot-v1-8k": {"context_length": 8192, "supports_function_calling": True},
            "moonshot-v1-32k": {"context_length": 32768, "supports_function_calling": True},
            "moonshot-v1-128k": {"context_length": 131072, "supports_function_calling": True},
            "kimi-k2-0711-preview": {"context_length": 131072, "supports_function_calling": True},
            "kimi-k2-turbo-preview": {"context_length": 131072, "supports_function_calling": True}
        }
    },
    "dashscope": {
        "adapter_class": ChatDashScopeOpenAIUnified,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "models": {
            "qwen-turbo": {"context_length": 8192, "supports_function_calling": True},
            "qwen-plus": {"context_length": 32768, "supports_function_calling": True},
            "qwen-plus-latest": {"context_length": 32768, "supports_function_calling": True},
            "qwen-max": {"context_length": 32768, "supports_function_calling": True},
            "qwen-max-latest": {"context_length": 32768, "supports_function_calling": True}
        }
    },
    "qianfan": {
        "adapter_class": ChatQianfanOpenAI,
        "base_url": "https://qianfan.baidubce.com/v2",
        "api_key_env": "QIANFAN_ACCESS_KEY",
        "models": {
            "ERNIE-Speed-8K": {"context_length": 8192, "supports_function_calling": True},
            "ERNIE-Lite-8K": {"context_length": 8192, "supports_function_calling": True}
        }
    },
    "custom_openai": {
        "adapter_class": ChatCustomOpenAI,
        "base_url": None,  # 将由用户配置
        "api_key_env": "CUSTOM_OPENAI_API_KEY",
        "models": {
            "gpt-3.5-turbo": {"context_length": 16384, "supports_function_calling": True},
            "gpt-4": {"context_length": 8192, "supports_function_calling": True},
            "gpt-4-turbo": {"context_length": 128000, "supports_function_calling": True},
            "gpt-4o": {"context_length": 128000, "supports_function_calling": True},
            "gpt-4o-mini": {"context_length": 128000, "supports_function_calling": True},
            "claude-3-haiku": {"context_length": 200000, "supports_function_calling": True},
            "claude-3-sonnet": {"context_length": 200000, "supports_function_calling": True},
            "claude-3-opus": {"context_length": 200000, "supports_function_calling": True},
            "claude-3.5-sonnet": {"context_length": 200000, "supports_function_calling": True},
            "gemini-pro": {"context_length": 32768, "supports_function_calling": True},
            "gemini-1.5-pro": {"context_length": 1000000, "supports_function_calling": True},
            "llama-3.1-8b": {"context_length": 128000, "supports_function_calling": True},
            "llama-3.1-70b": {"context_length": 128000, "supports_function_calling": True},
            "llama-3.1-405b": {"context_length": 128000, "supports_function_calling": True},
            "custom-model": {"context_length": 32768, "supports_function_calling": True}
        }
    }
}


def create_openai_compatible_llm(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> OpenAICompatibleBase:
    """创建OpenAI兼容LLM实例的统一工厂函数"""
    provider_info = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
    if not provider_info:
        raise ValueError(f"不支持的OpenAI兼容提供商: {provider}")

    adapter_class = provider_info["adapter_class"]

    # 如果调用未提供 base_url，则采用 provider 的默认值（可能为 None）
    if base_url is None:
        base_url = provider_info.get("base_url")

    # 仅当 provider 未内置 base_url（如 custom_openai）时，才将 base_url 传递给适配器，
    # 避免与适配器内部的 super().__init__(..., base_url=...) 冲突导致 "multiple values" 错误。
    init_kwargs = dict(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    if provider_info.get("base_url") is None and base_url:
        init_kwargs["base_url"] = base_url

    return adapter_class(**init_kwargs)


def test_openai_compatible_adapters():
    """快速测试所有适配器是否能被正确实例化（不发起真实请求）"""
    for provider, info in OPENAI_COMPATIBLE_PROVIDERS.items():
        cls = info["adapter_class"]
        try:
            if provider == "custom_openai":
                cls(model="gpt-3.5-turbo", api_key="test", base_url="https://api.openai.com/v1")
            elif provider == "qianfan":
                # 千帆新一代API仅需QIANFAN_API_KEY，格式: bce-v3/ALTAK-xxx/xxx
                cls(model="ernie-3.5-8k", api_key="bce-v3/test-key/test-secret")
            else:
                cls(model=list(info["models"].keys())[0], api_key="test")
            logger.info(f"✅ 适配器实例化成功: {provider}")
        except Exception as e:
            logger.warning(f"⚠️ 适配器实例化失败（预期或可忽略）: {provider} - {e}")


# NOTE FOR CONTRIBUTORS:
# To add a new OpenAI-compatible provider, follow these steps:
# 1) Create an adapter class by subclassing OpenAICompatibleBase (see ChatDeepSeekOpenAI/ChatDashScopeOpenAIUnified for examples)
# 2) Register the provider in OPENAI_COMPATIBLE_PROVIDERS with keys: adapter_class, base_url (if needed), api_key_env, and optional model metadata
# 3) Ensure the required API key environment variable is documented in docs/LLM_INTEGRATION_GUIDE.md and added to `.env.example`
# 4) If the provider requires a non-standard base_url, pass it via constructor or provider registry
# 5) Run the provided tests in this file (test_* functions) or add a similar smoke test for your provider
# Security: NEVER log raw API keys. Keep logging to high-level info only.

if __name__ == "__main__":
    test_openai_compatible_adapters()
