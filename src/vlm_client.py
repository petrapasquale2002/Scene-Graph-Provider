import os, sys, base64, requests, base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io import BytesIO
from typing import Any, Dict, Optional, Union, List
from PIL import Image, ImageDraw
try:    
    from google import genai
except ImportError:
    genai = None
try:
    from .base_client import BaseFoundationClient
except ImportError:
    from base_client import BaseFoundationClient

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
                return image_source # Return URL directly if provider supports it, or download and encode
            elif os.path.isfile(image_source):
                with open(image_source, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            else:
                 # Assume it's already base64 string if not file/url
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

    def __call__(self, text_prompt: Optional[str] = None, image: Union[str, bytes, Image.Image, None] = None, **kwargs) -> str:
        """Sends a vision-language request to the model."""
        force_json_response = kwargs.get("force_json_response", False)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        stream = kwargs.get("stream", self.stream)

        if self.provider == "groq":
            # Groq VLM accepts either remote URLs or inline base64 image data URLs.
            if self.provider == "groq":
                # Groq VLM accepts either remote URLs or inline base64 image data URLs.
                image_content = self._build_image_url_content(image, **kwargs)
  
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
                tools = kwargs.get("tools")
                if tools is not None:
                    params["tools"] = tools
                tool_choice = kwargs.get("tool_choice")
                if tool_choice is not None:
                    params["tool_choice"] = tool_choice
  
                response = self.client.chat.completions.create(**params)
                if hasattr(response, 'usage'):
                    self._update_metrics(response.usage.prompt_tokens, response.usage.completion_tokens)
                return response.choices[0].message.content

        elif self.provider in ["openai", "nebius"]:
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

            optional_params = [
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
            ]
            for param_name in optional_params:
                value = self._get_call_parameter(param_name, kwargs)
                if value is not None:
                    params[param_name] = value

            if force_json_response and "response_format" not in params:
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

        elif self.provider == "anthropic":

            raise NotImplementedError("VLMClient does not support Anthropic yet due to differences in image handling and API structure.")   
        
            base64_image = self._encode_image(image)
            # Anthropic needs media_type, assuming jpeg for simplicity or detect
            media_type = "image/jpeg"
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image,
                                },
                            },
                            {"type": "text", "text": text_prompt}
                        ],
                    }
                ],
            )
            self._update_metrics(response.usage.input_tokens, response.usage.output_tokens)
            return response.content[0].text

        elif self.provider == "gemini":

            raise NotImplementedError("VLMClient does not support Gemini yet due to differences in image handling and API structure.")
        
            # Gemini supports PIL images directly or bytes
            if isinstance(image, str):
                if image.startswith("http"):
                    # quick download for gemini
                    # genai SDK might handle urls if passed as Part/URI, but keeping it simple with requests
                    if requests:
                        response = requests.get(image)
                        img_data = Image.open(BytesIO(response.content))
                    else:
                        raise ImportError("Requests not installed for URL handling")
                elif os.path.isfile(image):
                    img_data = Image.open(image)
                else: 
                     # Base64 string
                     img_data = Image.open(BytesIO(base64.b64decode(image)))
            else:
                img_data = image

            config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[text_prompt, img_data],
                config=config
            )
            usage = response.usage_metadata
            self._update_metrics(usage.prompt_token_count, usage.candidates_token_count)
            return response.text

        else:
             raise NotImplementedError(f"Provider {self.provider} not supported for Vision.")


if __name__ == "__main__":

    from src.test.test_client import TestVLMClient

    model_parameters = {
        "model_name": "groq/llama4-scout-17b",
        'temperature': 0.0,
        'max_tokens': 512,
        'top_p': 1.0
    }

    vlm_client = TestVLMClient(**model_parameters)
    vlm_client.test_response_with_url_image()
    vlm_client.test_response_with_local_image(image_path='/home/petra/AI_VLM/kid.jpeg')
    vlm_client.test_response_with_local_video(video_path='/home/petra/AI_VLM/testFaceTracking.mp4')