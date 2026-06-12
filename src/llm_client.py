from litellm import completion


class LLMClient:
    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 1024, top_p: float = 1.0, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.extra = kwargs

    def __call__(self, system_message: str, user_message: str, **kwargs):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        response = completion(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            **self.extra,
            **kwargs,
        )

        try:
            return response.choices[0].message.content
        except Exception:
            return response
