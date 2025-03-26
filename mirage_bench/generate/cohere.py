# Make sure to have the newest Cohere SDK installed:
# pip install -U cohere
# Get your free API key from: www.cohere.com
from __future__ import annotations

import logging
import os
import time

import cohere

from .base import BaseClient

logger = logging.getLogger(__name__)


class CohereAPIClient(BaseClient):
    def __init__(
        self,
        model_name_or_path: str,
        api_key: str = None,
        wait: int = 10,
    ):
        self.deployment_name = model_name_or_path
        self.wait = wait
        logger.info(f"Initializing Cohere API Client: {model_name_or_path}")
        self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY") if api_key is None else api_key)

    def response(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        disable_logging: bool = False,
        **kwargs,
    ):
        try:
            if not disable_logging:
                generation_config = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                logger.info(f"Cohere generation config: {generation_config}")

            response = self.client.chat(
                model=self.deployment_name,  # model = "deployment_name".
                message=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.text

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
