from __future__ import annotations

import logging

import numpy as np
import ray
from datasets import Dataset
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class HFModel:
    def __init__(self, model_name_or_path: str, cache_dir: str | None = "./cache"):
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        logger.info(f"Initializing Huggingface Model: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=self.cache_dir, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, cache_dir=self.cache_dir, trust_remote_code=True
        )
        del model

    def _get_terminators_and_stop_strings(self) -> tuple[list[int], list[str]]:
        terminators, stop_strings = [], []
        if "llama-3" in self.model_name_or_path.lower():
            terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            stop_strings = ["<|eot_id|>"]
        return terminators, stop_strings


# Create a class to do batch inference.
class LLMPredictor:
    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        prompt_key: str = "prompt",
        max_model_len: int = 4096,
        max_num_seqs: int = 1,
        cache_dir: str = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        terminators: list[int] = None,
        stop_strings: list[str] = None,
        additional_keys: list[str] = None,
    ):
        self.prompt_key = prompt_key
        self.additional_keys = additional_keys
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop_token_ids=terminators if terminators else None,
            stop=stop_strings if stop_strings else None,
        )
        # Create an LLM.
        self.llm = LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tokenizer_mode="auto",
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_seq_len_to_capture=max_model_len,
            download_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )  # skip graph capturing for faster cold starts)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch[self.prompt_key], self.sampling_params)
        prompt: list[str] = []
        generated_text: list[str] = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(" ".join([o.text for o in output.outputs]))
        output_dict = {"output": generated_text, "query_id": batch["query_id"], "prompt": prompt}
        if self.additional_keys:
            for key in self.additional_keys:
                output_dict[key] = batch[key]
        return output_dict


class VLLMClient:
    def __init__(
        self,
        model_name_or_path: str,
        wait: int = 5,
        cache_dir: str | None = "./cache",
        prompt_key: str = "prompt",
        additional_keys: list[str] = None,
        tensor_parallel_size: int = 1,
        **kwargs,
    ):
        self.hf_model = HFModel(model_name_or_path, cache_dir)
        self.tensor_parallel_size = tensor_parallel_size
        terminators, stop_strings = self.hf_model._get_terminators_and_stop_strings()
        self.fn_kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "prompt_key": prompt_key,
            "additional_keys": additional_keys,
            "cache_dir": cache_dir,
            "terminators": terminators,
            "stop_strings": stop_strings,
        }
        self.fn_kwargs.update(kwargs)
        self.model_name_or_path = model_name_or_path

    # For tensor_parallel_size > 1, we need to create placement groups for vLLM
    # to use. Every actor has to have its own placement group.
    def scheduling_strategy_fn(self):
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{"GPU": 1, "CPU": 1}] * self.tensor_parallel_size,
            strategy="STRICT_PACK",
        )
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(pg, placement_group_capture_child_tasks=True))

    def batch_call(
        self,
        prompts: list[str],
        query_ids: list[str] = None,
        batch_size: int = 8,
        num_instances: int = 1,
        shards: int = 12,
        **kwargs,
    ):
        prompts_final = []
        for prompt in prompts:
            messages = [{"role": "user", "content": f"{prompt}"}]
            prompt_template = self.hf_model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts_final.append(prompt_template)

        logger.info("Coverted dataset into HF format ...")
        hf_dataset = {"query_id": query_ids, "prompt": prompts_final}
        hf_dataset = Dataset.from_dict(hf_dataset)
        logger.info(f"Example: {hf_dataset[0]}")

        # Convert the Huggingface dataset to Ray Data.
        ds = ray.data.from_huggingface(hf_dataset)

        # Apply batch inference for all input data.
        ds.repartition(shards, shuffle=False)

        resources_kwarg = {}
        if self.tensor_parallel_size == 1:
            # For tensor_parallel_size == 1, we simply set num_gpus=1.
            resources_kwarg["num_gpus"] = 1
        else:
            # Otherwise, we have to set num_gpus=0 and provide
            # a function that will create a placement group for
            # each instance.
            resources_kwarg["num_gpus"] = 0
            resources_kwarg["ray_remote_args_fn"] = self.scheduling_strategy_fn

        # log the function arguments passed to VLLM
        logger.info(f"Function arguments passed to VLLM: {self.fn_kwargs}")

        ds = ds.map_batches(
            LLMPredictor,
            # Pass the function arguments.
            fn_constructor_args=(self.model_name_or_path,),
            fn_constructor_kwargs=self.fn_kwargs,
            # Set the concurrency to the number of LLM instances.
            concurrency=num_instances,
            # Specify the batch size for inference.
            batch_size=batch_size,
            **resources_kwarg,
        )
        outputs = ds.take_all()
        return outputs
