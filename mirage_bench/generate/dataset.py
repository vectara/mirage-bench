from __future__ import annotations

import datasets
from transformers import AutoTokenizer

from .utils import load_jsonl


class HFDataset:
    def __init__(self, dataset_name: str, language: str | None, split: str | None, cache_dir: str | None = "./cache"):
        self.dataset_name = dataset_name
        self.language = language
        self.split = split
        self.cache_dir = cache_dir
        self.hf_dataset = None

    def load_dataset(self):
        self.hf_dataset = datasets.load_dataset(
            self.dataset_name, self.language, split=self.split, cache_dir=self.cache_dir
        )

    def load_dataset_local(self, input_dir: str):
        datasets_list = [datasets.Dataset.from_list(load_jsonl(input_filepath=filename)) for filename in input_dir]
        self.hf_dataset = datasets.concatenate_datasets(datasets_list)

    def format_prompt(self, prompt_key: str, output_key: str, tokenizer: AutoTokenizer, *args):
        prompts = []
        for idx, row in enumerate(self.hf_dataset):
            prompt = row[prompt_key]
            messages = [{"role": "user", "content": f"{prompt}"}]
            prompt_template = tokenizer.apply_chat_template(messages, args)
            prompts.append(prompt_template)

        self.hf_dataset = self.hf_dataset.add_column(output_key, prompts)

    def filter_dataset(self, filter_start: int = 0, filter_end: int = None):
        hf_dataset_length = len(self.hf_dataset)
        if filter_end is None:
            filter_end = hf_dataset_length
        elif filter_start > hf_dataset_length:
            return ValueError(
                f"Filter start {filter_start} is greater than the length of HF Dataset: {hf_dataset_length}. Exiting..."
            )
        elif filter_end > hf_dataset_length:
            filter_end = hf_dataset_length

        self.hf_dataset = self.hf_dataset.select(range(filter_start, filter_end))
