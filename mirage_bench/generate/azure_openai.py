from __future__ import annotations

import logging
import os
import time

import openai
from openai import AzureOpenAI

from .base import BaseClient

logger = logging.getLogger(__name__)


class AzureOpenAIClient(BaseClient):
    def __init__(
        self,
        model_name_or_path: str,
        endpoint: str = None,
        api_key: str = None,
        api_version: str = None,
        wait: int = 10,
    ):
        self.deployment_name = model_name_or_path
        self.wait = wait
        logger.info(f"Initializing OpenAI API Client: {model_name_or_path}")
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") if endpoint is None else endpoint,
            api_key=os.getenv("AZURE_OPENAI_API_KEY") if api_key is None else api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION") if api_version is None else api_version,
        )

    def response(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        n: int = 1,
        disable_logging: bool = False,
        **kwargs,
    ):
        try:
            if not disable_logging:
                generation_config = {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    "n": n,
                }
                logger.info(f"OpenAI generation config: {generation_config}")

            response = self.client.chat.completions.create(
                model=self.deployment_name,  # model = "deployment_name".
                messages=[{"role": "user", "content": f"{prompt}"}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                n=n,
                **kwargs,
            )
            return response.choices[0].message.content

        except openai.RateLimitError as e:
            retry_time = int(str(e).split("retry after")[1].split("second")[0].strip())
            logger.error(f"Rate limit exceeded. Retrying after waiting for {retry_time + 2} seconds...")
            time.sleep(retry_time + 2)
            return self.response(prompt, temperature, max_tokens, n, disable_logging, **kwargs)

        except openai.InternalServerError:
            logger.error(f"Internal server error. Retrying after waiting for {self.wait} seconds...")
            time.sleep(self.wait)
            return self.response(prompt, temperature, max_tokens, n, disable_logging, **kwargs)

    def __call__(
        self,
        prompts: list[str],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        n: int = 1,
        disable_logging: bool = False,
        **kwargs,
    ):
        responses = []
        for prompt in prompts:
            output = self.response(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                disable_logging=disable_logging,
                **kwargs,
            )
            responses.append(output)
        return responses
