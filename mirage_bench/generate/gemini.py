from __future__ import annotations

import logging
import os
import time

import google.generativeai as genai

from .base import BaseClient

logger = logging.getLogger(__name__)


class GeminiAPIClient(BaseClient):
    def __init__(self, model_name_or_path: str, wait: int = 60):
        super().__init__()
        """Gemini API Client"""
        self.wait = wait  # in seconds
        logger.info(f"Initializing Gemini API Client: {model_name_or_path}")
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model_name_or_path)

    def response(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        candidate_count: int = 1,
        disable_logging: bool = False,
        **kwargs,
    ):
        try:
            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "candidate_count": candidate_count,
            }
            if kwargs:
                generation_config.update(**kwargs)

            if not disable_logging:
                logger.info(f"Gemini generation config: {generation_config}")

            response = self.client.generate_content(prompt, generation_config=generation_config)
            return response.text

        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}")
            # Retry after waiting for a few seconds
            logger.info(f"Retrying after waiting for {self.wait} seconds...")
            time.sleep(self.wait)
            return self.response(prompt, temperature, max_tokens, candidate_count, disable_logging, **kwargs)

    def __call__(
        self,
        prompts: list[str],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        candidate_count: int = 1,
        disable_logging: bool = False,
        **kwargs,
    ):
        responses = []
        for prompt in prompts:
            output = self.response(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                candidate_count=candidate_count,
                disable_logging=disable_logging,
                **kwargs,
            )
            responses.append(output)
        return responses
