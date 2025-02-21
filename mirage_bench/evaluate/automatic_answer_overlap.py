from __future__ import annotations

import json
import logging
import os
import re

import ray
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from .util import ISO_TO_LANG, parse_text_wo_citation
from .vllm import LLMPredictor

logger = logging.getLogger(__name__)

DEFAULT_EVAL_PROMPT = """
You are an AI assistant. In the following task, you are given a Question, a RAG application's response, and a Ground-truth Answer referred to as 'Label' in {{language}}.
Assess how well the RAG application's response aligns with the Label, using the grading rubric below:\n\n
1: The response is not aligned with the Label or is off-topic; includes hallucination.
2: The response admits it cannot provide an answer or lacks context; honest.
3: The response is relevant but contains notable discrepancies or inaccuracies.
4: The response is acceptable, sufficient but not exhaustive.
5: The response is fully accurate and comprehensive, based on the Label.\n\n
Treat the Label as the definitive answer. Present your justification followed by your final score in the format: "[[score]]",
----
Example:
Justification: The response partially aligns with the label but with some discrepancies. Score: [[3]]
-----
Question in {{language}}:
{{question}}\n

Label in {{language}}:
{{label}}\n

RAG Application Response in {{language}}:
{{response}}\n
Treat the label as the definitive answer. Present your justification in English and followed by your final score in the format: "[[score]]",
"""


class AutomaticAnswerOverlapEvaluator:
    def __init__(
        self,
        language_code: str,
        model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        cache_dir: str = None,
        max_length: int = 8192,
        max_num_seqs: int = 1,
        dtype: str = "bfloat16",
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        trust_remote_code: bool = True,
        answer_regex: str = r"Answer:(.*?)$",
        metric_name: str = "answer_overlap",
    ):
        self.language_code = language_code
        self.answer_regex = answer_regex
        self.metric_name = metric_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        terminators, stop_strings = self._get_terminators_and_stop_strings(model_name_or_path, self.tokenizer)

        self.llm_predictor = LLMPredictor(
            model_name_or_path=model_name_or_path,
            max_model_len=max_length,
            max_num_seqs=max_num_seqs,
            cache_dir=cache_dir,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            terminators=terminators,
            stop_strings=stop_strings,
        )
        self.scores = None
        self.raw_predictions = None

    def _get_terminators_and_stop_strings(self, model_name, tokenizer):
        terminators, stop_strings = [], []
        if "llama-3" in model_name:
            terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            stop_strings = ["<|eot_id|>"]
        return terminators, stop_strings

    def postprocess(raw_prediction: str, regex: str) -> str:
        rating = 0
        coherence = re.search(regex, raw_prediction, re.DOTALL)
        if coherence:
            rating = coherence.group(1).strip()
            rating = rating.replace(":", "").replace("#", "").strip()
            try:
                return (int(rating), False)
            except ValueError:
                logger.warning(f"Rating is not an integer: {rating} Answer: {raw_prediction}")
                return (0, True)
        else:
            logger.warning(f"Rating not found in answer: {raw_prediction}")
            return (0, True)

    def save(self, output_file: str):
        """Save the scores (and raw_output) to a file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as fout:
            for query_id in self.scores:
                data = {"query_id": query_id, self.metric_name: self.scores[query_id][self.metric_name]}
                if self.raw_predictions:
                    data["raw_output"] = self.raw_predictions[query_id]
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    def evaluate(
        self,
        predictions: dict[str, str],
        reference_predictions: dict[str, str],
        documents: dict[str, dict[str, str]],
        queries: dict[str, str],
        prompt: str = DEFAULT_EVAL_PROMPT,
        batch_size: int = 128,
        num_gpus: int = 1,
        concurrency: int = 4,
        postprocess_regex: str = r"Score:(.*?)$",  # regex to extract the rating from the raw prediction
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate the model on the given predictions and return the scores.
        Args:
            predictions (dict): The predictions of the model, where the key is the query_id and the value is the RAG answer.
            documents (dict): The document_id and document text for each query_id.
            batch_size (int): The batch size for computing the NLI scores.
        Returns:
            dict: The entailment, contrdiction and neutral scores for each query_id.
        """
        self.scores = {query_id: {self.metric_name: 0} for query_id in documents}

        prompts = []
        for query_id in tqdm(documents, desc="Processing queries", total=len(documents)):
            doc_ids = documents[query_id]  # get the doc_ids for the query_id
            rag_answer = parse_text_wo_citation(predictions[query_id], regex=self.answer_regex, doc_ids=doc_ids)
            reference_rag_answer = parse_text_wo_citation(
                reference_predictions[query_id], regex=self.answer_regex, doc_ids=doc_ids
            )
            query_text = queries[query_id]
            prompt = prompt.format(
                language=ISO_TO_LANG[self.language_code].capitalize(),
                question=query_text.strip(),
                label=reference_rag_answer.strip(),
                response=rag_answer,
            )
            messages = [{"role": "user", "content": f"{prompt}"}]
            prompt_template = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_template)

        hf_dataset = {"prompt": prompts, "query_id": list(documents.keys())}
        hf_dataset = Dataset.from_dict(hf_dataset)

        # Convert the Huggingface dataset to Ray Data.
        ds = ray.data.from_huggingface(hf_dataset)

        # Apply batch inference for all input data.
        ds = ds.repartition(12, shuffle=False)

        ds = ds.map_batches(
            LLMPredictor,
            # Set the concurrency to the number of LLM instances.
            concurrency=concurrency,
            # Specify the number of GPUs required per LLM instance.
            # NOTE: Do NOT set `num_gpus` when using vLLM with tensor-parallelism
            # (i.e., `tensor_parallel_size`).
            num_gpus=num_gpus,
            # Specify the batch size for inference.
            batch_size=batch_size,
            zero_copy_batch=True,
        )

        # NOTE: This is for local testing and debugging. For production use case,
        # one should write full result out as shown below.
        outputs = ds.take_all()

        for output in outputs:
            query_id = output["query_id"]
            raw_prediction = output["output"]
            (rating, _) = self.postprocess(raw_prediction=raw_prediction, regex=postprocess_regex)
            self.scores[query_id[self.metric_name]] = rating
            self.raw_predictions[query_id] = raw_prediction

        ### Logging the average scores
        logger.info("Averaging the scores achieved by the model ...")
        logger.info("-" * 50)
        avg_score = sum([self.scores[query_id][self.metric_name] for query_id in documents]) / len(documents)
        logger.info(f"Avg Answer Overlap Score: {avg_score:8.4f}")
        logger.info("-" * 50)

        return self.scores
