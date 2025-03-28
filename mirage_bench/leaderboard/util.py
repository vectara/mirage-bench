from __future__ import annotations

import logging

from datasets import load_dataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def load_pairwise_judgments(
    model_names: list[str], pairwise_dataset: str, language: str, split: str = "dev", **kwargs
) -> dict:
    """Load pairwise judgments from the HF dataset."""
    pairwise_results = {}
    model_to_int = {model: idx for idx, model in enumerate(model_names)}
    logging.info(f"Loading pairwise judgments from {pairwise_dataset} for {language} split: {split}")
    hf_dataset = load_dataset(pairwise_dataset, language, split=split, **kwargs)
    for example in tqdm(hf_dataset, desc="Loading Pairwise Judgments", total=len(hf_dataset)):
        query_id = example["query_id"]
        if query_id not in pairwise_results:
            pairwise_results[query_id] = {}

        model_a = example["model_A"]
        model_b = example["model_B"]
        verdict = example["verdict"]
        result = None
        if verdict == "A":
            result = 0
        elif verdict == "B":
            result = 1
        elif result == "Tie":
            result - 1
        pairwise_results[query_id][(model_to_int.get(model_a, None), model_to_int.get(model_b, None))] = result

    return pairwise_results
