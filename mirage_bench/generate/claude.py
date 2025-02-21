from __future__ import annotations

import logging
import os
import time

import anthropic
from anthropic import Anthropic

from .base import BaseClient

logger = logging.getLogger(__name__)


class ClaudeAPIClient(BaseClient):
    def __init__(self, model_name_or_path: str = None, wait: int = 60):
        """Claude API Client"""
        self.deployment_name = model_name_or_path
        self.wait = wait
        logger.info(f"Initializing Claude API Client: {model_name_or_path}")
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def response(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024, disable_logging: bool = False, **kwargs
    ):
        try:
            if not disable_logging:
                generation_config = {"max_tokens": max_tokens, "temperature": temperature}
                logger.info(f"Claude generation config: {generation_config}")

            message = self.client.messages.create(
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": f"{prompt}"}],
                model=self.deployment_name,
            )
            return (message.content[0]).text

        except anthropic.APIConnectionError as e:
            logger.error("The server could not be reached")
            logger.error(e.__cause__)  # an underlying Exception, likely raised within httpx.
            logger.info(f"Retrying after waiting for {self.wait} seconds...")
            time.sleep(self.wait)  # Retry after waiting for a few seconds
            return self.response(prompt, temperature, max_tokens, disable_logging, **kwargs)

        except anthropic.RateLimitError:
            logger.error("A 429 status code was received; we should back off a bit.")
            logger.info(f"Retrying after waiting for {self.wait} seconds...")
            time.sleep(self.wait)
            return self.response(prompt, temperature, max_tokens, disable_logging, **kwargs)

        except anthropic.APIStatusError as e:
            logger.error("Another non-200-range status code was received")
            logger.error(e.status_code)
            logger.error(e.response)
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
