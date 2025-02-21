from __future__ import annotations

import numpy as np
from vllm import LLM, SamplingParams


# Create a class to do batch inference.
class LLMPredictor:
    def __init__(
        self,
        model_name_or_path: str,
        max_model_len: int = 4096,
        max_num_seqs: int = 1,
        cache_dir: str = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        terminators: list[int] = None,
        stop_strings: list[str] = None,
    ):
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop_token_ids=terminators if terminators else None,
            stop=stop_strings if stop_strings else None,
        )

        # Create an LLM.
        self.llm = LLM(
            model=model_name_or_path,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_seq_len_to_capture=max_model_len,
            download_dir=cache_dir,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )  # skip graph capturing for faster cold starts)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch["prompt"], self.sampling_params)
        prompt = []
        generated_text = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(" ".join([o.text for o in output.outputs]))

        return {"output": generated_text, "query_id": batch["query_id"]}
