import os, base64, base64, json, re
from io import BytesIO
from typing import Any, Dict, Optional, Type, Union
from PIL import Image, ImageDraw
try:    
    from google import genai
except ImportError:
    genai = None
try:
    from .base_client import BaseFoundationClient
except ImportError:
    from src.base_client import BaseFoundationClient
try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None

class VLMClient(BaseFoundationClient):
    """
    Client for Vision-Language tasks.
    """

    def __init__(self, **model_parameters):
        super().__init__(**model_parameters)

    def _get_call_parameter(self, name: str, kwargs: Dict[str, Any], default: Any = None) -> Any:
        if name in kwargs:
            return kwargs[name]
        return self.model_parameters.get(name, default)

    def _encode_image(self, image_source: Union[str, bytes, Image.Image]) -> str:
        """Encodes image to base64 string."""
        if isinstance(image_source, Image.Image):
            buffered = BytesIO()
            image_source.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(image_source, bytes):
            return base64.b64encode(image_source).decode('utf-8')
        elif isinstance(image_source, str):
            if image_source.startswith("http"):
                return image_source
            elif os.path.isfile(image_source):
                with open(image_source, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            else:
                 return image_source
        return ""

    def _build_image_url_content(self, image: Union[str, bytes, Image.Image], **kwargs) -> Dict[str, Any]:
        if image is None:
            raise ValueError("image must be provided when messages are not passed.")

        image_url = {}
        if isinstance(image, str) and image.startswith("http"):
            image_url["url"] = image
        else:
            mime_type = kwargs.get("image_mime_type", "image/jpeg")
            image_url["url"] = f"data:{mime_type};base64,{self._encode_image(image)}"

        image_detail = kwargs.get("image_detail")
        if image_detail is not None:
            image_url["detail"] = image_detail

        return {
            "type": "image_url",
            "image_url": image_url
        }

    @staticmethod
    def _extract_json(raw: str) -> str:
        """
        Robustly extract a JSON object or array from a raw model response.

        Tries three strategies in order:
          1. Direct parse — the response is already valid JSON.
          2. Strip markdown fences (```json ... ```) then parse.
          3. Regex scan — find the first '{...}' or '[...]' block and parse it.

        Returns the first valid JSON string found, or raises ValueError if none found.
        """
        if not raw or not raw.strip():
            raise ValueError("Empty response from model")

        # Strategy 1: direct parse
        try:
            json.loads(raw)
            return raw
        except json.JSONDecodeError:
            pass

        # Strategy 2: strip markdown code fences
        stripped = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r'```\s*$', '', stripped.strip())
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

        # Strategy 3: find the first JSON object or array block
        for pattern in (r'(\{.*\})', r'(\[.*\])'):
            match = re.search(pattern, raw, flags=re.DOTALL)
            if match:
                candidate = match.group(1)
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

        raise ValueError(
            f"Could not extract valid JSON from model response. "
            f"First 200 chars: {raw[:200]!r}"
        )

    def _draw_bbs(self, bbs: list, image: Union[str, Image.Image], print: bool = False):
        if isinstance(image, str):
            image = Image.open(image)

        image = image.copy()
        draw = ImageDraw.Draw(image)

        for bb in bbs:
            coordinates = bb.get("coordinates", bb)
            x_min = coordinates["x_min"]
            y_min = coordinates["y_min"]
            x_max = coordinates["x_max"]
            y_max = coordinates["y_max"]
            label = bb.get("label", "")

            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            if label:
                draw.text((x_min, max(0, y_min - 12)), label, fill="red")

        if print:
            image.show()
            return

        return image

    def _call_groq(self, text_prompt: Optional[str], image: Union[str, bytes, Image.Image, None], force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)

        base64_image = self._encode_image(image)

        if isinstance(image, str) and image.startswith("http"):
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": image
                }
            }
        else:
            # base64_image covers local files, bytes, PIL images, and raw base64 strings
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }

        params = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        image_content,
                    ]
                }
            ],
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if force_json and forced_json_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": forced_json_schema.__name__,
                    "schema": forced_json_schema.model_json_schema()
                }
            }
        # Note: Groq VLMs (e.g. qwen 3.6) don't support response_format json_object.
        # JSON output is enforced via the prompt instead.

        response = self.client.chat.completions.create(**params)
        if hasattr(response, 'usage'):
            self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content

    def _call_openai(self, text_prompt: Optional[str], image: Union[str, bytes, Image.Image, None], force_json: bool = False, **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        stream = kwargs.get("stream", self.stream)

        messages = kwargs.get("messages")
        if messages is None:
            if text_prompt is None:
                raise ValueError("text_prompt must be provided when messages are not passed.")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        self._build_image_url_content(image, **kwargs),
                    ],
                }
            ]

        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

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

        if force_json and "response_format" not in params:
            params["response_format"] = {"type": "json_object"}

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
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end="", flush=True)
            print()
            return full_response

        if hasattr(response, 'usage'):
            self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content

    def _call_nebius(self, text_prompt: Optional[str], image: Union[str, bytes, Image.Image, None], force_json: bool = False, **kwargs) -> str:
        return self._call_openai(text_prompt, image, force_json=force_json, **kwargs)

    def _call_anthropic(self, text_prompt: Optional[str], image: Union[str, bytes, Image.Image, None], **kwargs) -> str:
        raise NotImplementedError("VLMClient does not support Anthropic yet due to differences in image handling and API structure.")   

    def _call_gemini(self, text_prompt: Optional[str], image: Union[str, bytes, Image.Image, None], **kwargs) -> str:
        raise NotImplementedError("VLMClient does not support Gemini yet due to differences in image handling and API structure.")

    def __call__(self, text_prompt: Optional[str] = None, image: Union[str, bytes, Image.Image, None] = None, force_json: bool = False, forced_json_schema: Optional[Type['BaseModel']] = None, **kwargs) -> str:
        """Sends a vision-language request to the model."""
        if self.provider == "groq":
            return self._call_groq(text_prompt, image, force_json=force_json, forced_json_schema=forced_json_schema, **kwargs)
        elif self.provider == "openai":
            return self._call_openai(text_prompt, image, force_json=force_json, **kwargs)
        elif self.provider == "nebius":
            return self._call_nebius(text_prompt, image, force_json=force_json, **kwargs)
        elif self.provider == "anthropic":
            return self._call_anthropic(text_prompt, image, **kwargs)
        elif self.provider == "gemini":
            return self._call_gemini(text_prompt, image, **kwargs)
        else:
             raise NotImplementedError(f"Provider {self.provider} not supported for Vision.")


if __name__ == "__main__":
        
        use_nebius = True
        use_groq = False

        if use_nebius:
            model_parameters = {
                "model_name": "nebius/qwen3-2.5-70b",
                'temperature': 0.7,
                'max_tokens': 2048,
                'top_p': 0.9,
            }
        elif use_groq:
            model_parameters = {
                "model_name": "groq/llama4-scout-17b",
                'temperature': 0.7,
                'max_tokens': 2048,
                'top_p': 0.9
            }
             
        vlm = VLMClient(**model_parameters)
        task = 'Find all the faces in the image. If there are specific known, please label them with their names.'
        
        bb_prompt = """
        Task: {task}.
        The image is provided in the size of {pixels_width} x {pixels_height}.
        Strictly use the following json format for the response, avoid any additional text or explanation.

        {{
        "bounding_boxes": [
            {{
                "label": "detection-label",
                "x_min": top-left-x-pixel,
                "y_min": top-left-y-pixel,
                "x_max": bottom-right-x-pixel,
                "y_max": bottom-right-y-pixel
            }}, 
            ]
        }}
        """