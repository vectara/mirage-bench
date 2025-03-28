from __future__ import annotations

import logging
import os
import time

import requests

from .base import BaseClient

logger = logging.getLogger(__name__)


class AnyScaleAPIClient(BaseClient):
    def __init__(self, model_name_or_path: str, api_key: str, api_base_url: str, wait: int = 60):
        self.deployment_name = model_name_or_path
        self.api_key = os.getenv("ANYSCALE_API_KEY") if api_key is None else api_key
        self.api_base_url = os.getenv("ANYSCALE_BASE_URL") if api_base_url is None else api_base_url
        self.url = f"{self.api_base_url}/chat/completions"
        self.s = requests.Session()
        self.wait = wait

    def response(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024, disable_logging: bool = False, **kwargs
    ):
        try:
            generation_config = {
                "model_name": self.deployment_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if not disable_logging:
                logger.info(f"AnyScale generation config: {generation_config}")

            generation_config.update({"messages": [{"role": "user", "content": f"{prompt}"}]})
            with self.s.post(
                self.url, headers={"Authorization": f"Bearer {self.api_key}"}, json=generation_config
            ) as resp:
                return resp.json()["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}")
            # Retry after waiting for a few seconds
            logger.info(f"Retrying after waiting for {self.wait} seconds...")
            time.sleep(self.wait)
            return self.response(prompt, temperature, max_tokens, disable_logging, **kwargs)

    def __call__(
        self,
        prompts: list[str],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        disable_logging: bool = False,
        **kwargs,
    ):
        responses = []
        for prompt in prompts:
            output = self.response(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                disable_logging=disable_logging,
                **kwargs,
            )
            responses.append(output)
        return responses
