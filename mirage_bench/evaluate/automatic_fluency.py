from __future__ import annotations

import json
import logging
import os
import re

import numpy as np
import ray
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .util import ISO_TO_LANG

logger = logging.getLogger(__name__)

FLUENCY_PROMPT = """
You will be given one summary written for a question and documents from Wikipedia in {{language}}.
Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions carefully. Please keep this
document open while reviewing, and refer to it as needed.\n\n
Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences. We align this dimension with
the DUC quality question of structure and coherence whereby 'the summary should be
well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.'\n\n
Evaluation Steps:
1. Read the question and Wikipedia documents in {{language}} carefully and identify the main topic and key points.
2. Read the summary and check whether it answers the question. Check if the summary covers the main
topic and key points required to answer the question, and if it presents them in a clear and logical order.
3. Assign a rating for coherence on a scale of 1 to 5 and provide an explanation, where 1 is the lowest and 5 is the highest
based on the Evaluation Criteria.\n\n
Example:
Question in {{language}}:
{{Question}}\n
Documents in {{language}}:
{{Documents}}\n
Summary:
{{Summary}}\n
Rate the coherence of the summary on a scale of 1 to 5 and provide an explanation for your rating. Please use the format of: ##Rating: {rating} ##Explanation: {explanation}.
"""


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


class AutomaticFluencyEvaluator:
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
    ):
        self.language_code = language_code
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

    def postprocess(raw_prediction: str) -> str:
        rating = 0
        coherence = re.search(r"Rating(.*?)Explanation:", raw_prediction, re.DOTALL)
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
                data = {"query_id": query_id, "fluency": self.scores[query_id]["fluency-score"]}
                if self.raw_predictions:
                    data["raw_output"] = self.raw_predictions[query_id]
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    def evaluate(
        self,
        predictions: dict[str, str],
        documents: dict[str, dict[str, str]],
        queries: dict[str, str],
        batch_size: int = 128,
        num_gpus: int = 1,
        concurrency: int = 4,
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
        self.scores = {query_id: {"fluency_score": 0} for query_id in documents}

        prompts = []
        for query_id in tqdm(documents, desc="Processing queries", total=len(documents)):
            rag_answer = predictions[query_id].replace("##Reason:", "Reason:").replace("##Answer:", "Answer:")
            doc_ids = documents[query_id]
            document_text = ""
            for doc_id in doc_ids:
                document_text += f"[{doc_id}]: {documents[query_id][doc_id]}\n"
            query_text = queries[query_id]
            prompt = FLUENCY_PROMPT.format(
                language=ISO_TO_LANG[self.language_code].capitalize(),
                Question=query_text.strip(),
                Documents=document_text.strip(),
                Summary=rag_answer,
            )
            messages = [{"role": "user", "content": f"{prompt}"}]
            prompt_template = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_template)

        hf_dataset = {"prompt": prompts, "query_id": list(document_text.keys())}
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

        for idx, output in enumerate(outputs):
            query_id = output["query_id"]
            raw_prediction = output["output"]
            (rating, error) = self.postprocess(raw_prediction=raw_prediction)
            self.scores[query_id["fluency_score"]] = rating
            self.raw_predictions[query_id] = raw_prediction

        ### Logging the average scores
        logger.info("Averaging the scores achieved by the model ...")
        logger.info("-" * 50)
        avg_fluency_score = sum([self.scores[query_id]["fluency-score"] for query_id in documents]) / len(documents)
        logger.info(f"Avg Fluency Score: {avg_fluency_score:8.4f}")
        logger.info("-" * 50)

        return self.scores
