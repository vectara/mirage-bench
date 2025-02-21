from __future__ import annotations

from tqdm import tqdm


class BaseClient:
    def __init__(self, **kwargs):
        pass

    def __call__(self, prompts: list[str], **kwargs):
        raise NotImplementedError()

    def response(self, prompt: str, **kwargs):
        raise NotImplementedError()

    def batch_call(self, prompts: list[str], batch_size: int = 1, **kwargs) -> list[str]:
        batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

        results = []
        for i, batch in enumerate(tqdm(batches, desc="Collecting responses", leave=False)):
            responses = self.__call__(batch, **kwargs)
            results.extend(responses)
        return results
