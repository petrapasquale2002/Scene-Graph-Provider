from typing import Any, Dict, Optional, Type
try:
    from .base_client import BaseFoundationClient
except ImportError:
    from src.base_client import BaseFoundationClient
try:
    from google import genai
except ImportError:
    genai = None
try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None


class LLMClient(BaseFoundationClient):
    """
    Client for Text-to-Text interaction.
    """

    def __init__(self, **model_parameters):
        super().__init__(**model_parameters)

    def _get_call_parameter(self, name: str, kwargs: Dict[str, Any], default: Any = None) -> Any:
        if name in kwargs:
            return kwargs[name]
        return self.model_parameters.get(name, default)

    def _build_messages(self, user_message: Optional[str], system_message: str, messages: Optional[list]) -> list:
        if messages is not None:
            return messages

        built_messages = []
        if system_message:
            built_messages.append({"role": "system", "content": system_message})
        if user_message is not None:
            built_messages.append({"role": "user", "content": user_message})
        return built_messages

    def _handle_stream(self, response):
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end="", flush=True)
        print()
        return full_response

    def _update_from_openai_usage(self, response):
        if hasattr(response, "usage"):
            self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)

    def _call_groq(self, user_message: Optional[str], system_message: str, force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        stream = kwargs.get("stream", self.stream)
        full_content = kwargs.get("full_content", False)
        messages = self._build_messages(user_message, system_message, kwargs.get("messages"))

        if not messages:
            raise ValueError("Either user_message or messages must be provided.")

        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "max_completion_tokens": max_tokens,
        }
        if force_json and forced_json_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": forced_json_schema.__name__,
                    "schema": forced_json_schema.model_json_schema()
                }
            }
        elif force_json:
            params["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**params)
        
        if stream:
            return self._handle_stream(response)
        self._update_from_openai_usage(response)
        if full_content:
            return response
        return response.choices[0].message.content

    def _call_openai(self, user_message: Optional[str], system_message: str, force_json: bool = False, **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        stream = kwargs.get("stream", self.stream)
        full_content = kwargs.get("full_content", False)
        messages = self._build_messages(user_message, system_message, kwargs.get("messages"))

        if not messages:
            raise ValueError("Either user_message or messages must be provided.")

        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "max_tokens": max_tokens,
        }
        if force_json:
            params["response_format"] = {"type": "json_object"}

        for param_name in [
            "n",
            "stream_options",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "user",
            "response_format",
        ]:
            value = self._get_call_parameter(param_name, kwargs)
            if value is not None:
                params[param_name] = value

        extra_body = self._get_call_parameter("extra_body", kwargs)
        guided_json = self._get_call_parameter("guided_json", kwargs)
        top_k = self._get_call_parameter("top_k", kwargs)
        if extra_body is not None:
            extra_body = dict(extra_body)
        elif guided_json is not None or top_k is not None:
            extra_body = {}
        if guided_json is not None:
            extra_body["guided_json"] = guided_json
        if top_k is not None:
            extra_body["top_k"] = top_k
        if extra_body is not None:
            params["extra_body"] = extra_body

        response = self.client.chat.completions.create(**params)
        if stream:
            return self._handle_stream(response)
        self._update_from_openai_usage(response)
        if full_content:
            return response
        return response.choices[0].message.content

    def _call_nebius(self, user_message: Optional[str], system_message: str, force_json: bool = False, **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        stream = kwargs.get("stream", self.stream)
        full_content = kwargs.get("full_content", False)
        messages = self._build_messages(user_message, system_message, kwargs.get("messages"))

        if not messages:
            raise ValueError("Either user_message or messages must be provided.")

        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "max_tokens": max_tokens,
        }
        if force_json:
            params["response_format"] = {"type": "json_object"}

        for param_name in [
            "n",
            "stream_options",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "user",
            "response_format",
        ]:
            value = self._get_call_parameter(param_name, kwargs)
            if value is not None:
                params[param_name] = value

        extra_body = self._get_call_parameter("extra_body", kwargs)
        guided_json = self._get_call_parameter("guided_json", kwargs)
        top_k = self._get_call_parameter("top_k", kwargs)
        if extra_body is not None:
            extra_body = dict(extra_body)
        elif guided_json is not None or top_k is not None:
            extra_body = {}
        if guided_json is not None:
            extra_body["guided_json"] = guided_json
        if top_k is not None:
            extra_body["top_k"] = top_k
        if extra_body is not None:
            params["extra_body"] = extra_body

        response = self.client.chat.completions.create(**params)
        if stream:
            return self._handle_stream(response)
        self._update_from_openai_usage(response)
        if full_content:
            return response
        return response.choices[0].message.content

    def _call_anthropic(self, user_message: Optional[str], system_message: str, force_json: bool = False, **kwargs) -> str:
        raise NotImplementedError("LLMClient does not support Anthropic yet due to differences in system message handling.")

    def _call_gemini(self, user_message: Optional[str], system_message: str, force_json: bool = False, **kwargs) -> str:
        raise NotImplementedError("LLMClient does not support Gemini yet due to differences in system message handling.")

    def __call__(self, user_message: Optional[str] = None, system_message: str = "You are a helpful assistant.", force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        if self.provider == "groq":
            return self._call_groq(user_message, system_message, force_json, forced_json_schema=forced_json_schema, **kwargs)
        if self.provider == "openai":
            return self._call_openai(user_message, system_message, force_json, **kwargs)
        if self.provider == "nebius":
            return self._call_nebius(user_message, system_message, force_json, **kwargs)
        if self.provider == "anthropic":
            return self._call_anthropic(user_message, system_message, force_json, **kwargs)
        if self.provider == "gemini":
            return self._call_gemini(user_message, system_message, force_json, **kwargs)
        raise NotImplementedError(f"Provider {self.provider} not implemented.")


if __name__ == "__main__":

    from src.test.test_client import TestLLMClient  # type: ignore[reportMissingImports]

    model_parameters = {
        "model_name": "Nebius/openai-oss-20b",
        'temperature': 1.5,
        'max_tokens': 2048,
        'top_p': 0.9
    }

    llm_client = TestLLMClient(
        **model_parameters
    )

    llm_client.test_response()