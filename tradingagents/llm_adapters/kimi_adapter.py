"""
Moonshot AI 大模型 (Kimi) 适配器
为 TradingAgents 提供 Moonshot AI 大模型的 LangChain 兼容接口
"""

import os
import json
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator, Sequence
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr
import openai  # Moonshot 使用兼容 OpenAI 的 API 格式

from ..config.config_manager import token_tracker

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')



class ChatKimi(BaseChatModel):
    """Moonshot AI 大模型的 LangChain 适配器"""
    
    # 模型配置
    model: str = Field(default="moonshot-v1-8k", description="Kimi 模型名称")
    api_key: Optional[SecretStr] = Field(default=None, description="Moonshot API 密钥")
    base_url: str = Field(default="https://api.moonshot.cn/v1", description="API 基础 URL")
    temperature: float = Field(default=0.1, description="生成温度")
    max_tokens: int = Field(default=2000, description="最大生成token数")
    top_p: float = Field(default=0.9, description="核采样参数")
    
    # 内部属性
    _client: Any = None
    
    def __init__(self, **kwargs):
        """初始化 Moonshot AI 客户端"""
        super().__init__(**kwargs)
        
        # 设置API密钥
        api_key = self.api_key
        if api_key is None:
            api_key = os.getenv("MOONSHOT_API_KEY")
        
        if api_key is None:
            raise ValueError(
                "Moonshot API key not found. Please set MOONSHOT_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # 配置客户端
        if isinstance(api_key, SecretStr):
            api_key_str = api_key.get_secret_value()
        else:
            api_key_str = api_key
            
        # 使用 OpenAI 兼容客户端
        self._client = openai.OpenAI(
            api_key=api_key_str,
            base_url=self.base_url
        )
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "kimi"
    
    def _convert_messages_to_kimi_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """将 LangChain 消息格式转换为 Moonshot 格式"""
        kimi_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                # 默认作为用户消息处理
                role = "user"
            
            content = message.content
            if isinstance(content, list):
                # 处理多模态内容，目前只提取文本
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                content = text_content
            
            kimi_messages.append({
                "role": role,
                "content": str(content)
            })
        
        return kimi_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成聊天回复"""
        
        # 转换消息格式
        kimi_messages = self._convert_messages_to_kimi_format(messages)
        
        # 准备请求参数
        request_params = {
            "model": self.model,
            "messages": kimi_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        
        # 添加停止词
        if stop:
            request_params["stop"] = stop
        
        # 合并额外参数
        request_params.update(kwargs)
        
        try:
            # 调用 Moonshot API
            response = self._client.chat.completions.create(**request_params)
            
            if response.choices and response.choices[0].message:
                # 解析响应
                message_content = response.choices[0].message.content
                
                # 提取token使用量信息
                input_tokens = 0
                output_tokens = 0
                
                # Moonshot API响应中包含usage信息
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                
                # 记录token使用量
                if input_tokens > 0 or output_tokens > 0:
                    try:
                        # 生成会话ID（如果没有提供）
                        session_id = kwargs.get('session_id', f"kimi_{hash(str(messages))%10000}")
                        analysis_type = kwargs.get('analysis_type', 'stock_analysis')
                        
                        # 使用TokenTracker记录使用量
                        token_tracker.track_usage(
                            provider="kimi",
                            model_name=self.model,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            session_id=session_id,
                            analysis_type=analysis_type
                        )
                    except Exception as track_error:
                        # 记录失败不应该影响主要功能
                        logger.info(f"Token tracking failed: {track_error}")
                
                # 创建 AI 消息
                ai_message = AIMessage(content=message_content)
                
                # 创建生成结果
                generation = ChatGeneration(message=ai_message)
                
                return ChatResult(generations=[generation])
            else:
                raise Exception(f"Moonshot API error: No valid response received")
                
        except Exception as e:
            raise Exception(f"Error calling Moonshot API: {str(e)}")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成聊天回复"""
        # 目前使用同步方法，后续可以实现真正的异步
        return self._generate(messages, stop, run_manager, **kwargs)
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, BaseTool]],
        **kwargs: Any,
    ) -> "ChatKimi":
        """绑定工具到模型"""
        # Moonshot 支持 OpenAI 兼容的工具调用格式
        formatted_tools = []
        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "description"):
                # 这是一个 BaseTool 实例
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": getattr(tool, "args_schema", {})
                    }
                })
            elif isinstance(tool, dict):
                if "type" not in tool:
                    tool = {"type": "function", "function": tool}
                formatted_tools.append(tool)
            else:
                # 尝试转换为 OpenAI 工具格式
                try:
                    openai_tool = convert_to_openai_tool(tool)
                    if "type" not in openai_tool:
                        openai_tool = {"type": "function", "function": openai_tool}
                    formatted_tools.append(openai_tool)
                except Exception:
                    pass

        # 创建新实例，保存工具信息
        new_instance = self.__class__(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            **kwargs
        )
        new_instance._tools = formatted_tools
        return new_instance

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回标识参数"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }


# 支持的模型列表
KIMI_MODELS = {
    # Moonshot AI 系列 生成模型
    "moonshot-v1-8k": {
        "description": "Moonshot v1 8K 版本 - 适合日常对话和简单分析",
        "context_length": 8000,
        "recommended_for": ["快速任务", "日常对话", "简单分析"]
    },
    "moonshot-v1-32k": {
        "description": "Moonshot v1 32K 版本 - 平衡性能和上下文长度",
        "context_length": 32000,
        "recommended_for": ["复杂分析", "中等长度文档", "深度思考"]
    },
    "moonshot-v1-128k": {
        "description": "Moonshot v1 128K 版本 - 超长上下文支持",
        "context_length": 128000,
        "recommended_for": ["长文档分析", "大量数据处理", "复杂推理"]
    },
    # kimi-k2 模型
    "kimi-k2-0711-preview": {
        "description": "Kimi K2 0711 预览版 - 131K 模型，支持长上下文",
        "context_length": 131072,
        "recommended_for": ["长文档分析", "大量数据处理", "复杂推理"]
    },
    "kimi-k2-turbo-preview": {
        "description": "Kimi K2 的高速版 - 131K 模型，支持长上下文",
        "context_length": 131072,
        "recommended_for": ["长文档分析", "大量数据处理", "复杂推理"]
    }
}


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """获取可用的 Kimi 模型列表"""
    return KIMI_MODELS


def create_kimi_llm(
    model: str = "moonshot-v1-32k",
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    **kwargs
) -> ChatKimi:
    """创建 Kimi LLM 实例的便捷函数"""
    
    return ChatKimi(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )