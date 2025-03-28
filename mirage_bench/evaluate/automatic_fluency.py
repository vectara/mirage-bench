from __future__ import annotations

import json
import logging
import os
import re

from tqdm import tqdm

from mirage_bench.generate import VLLMClient

from .util import ISO_TO_LANG

logger = logging.getLogger(__name__)

DEFAULT_FLUENCY_PROMPT = """
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
{{question}}\n
Documents in {{language}}:
{{documents}}\n
Summary:
{{rag_answer}}\n
Provide an explanation and rate the coherence of the summary on a scale of 1 to 5 and provide an explanation for your rating.
Please use the format of: ##Explanation: <explanation> ##Rating: <rating>.
"""


class AutomaticFluencyEvaluator:
    def __init__(
        self,
        language_code: str,
        model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        tensor_parallel_size: int = 1,
        cache_dir: str = None,
        max_length: int = 8192,
        max_num_seqs: int = 1,
        dtype: str = "bfloat16",
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        trust_remote_code: bool = True,
        metric_name: str = "fluency_score",
        prompt_key: str = "prompt",
        additional_keys: list[str] = None,
    ):
        self.language_code = language_code
        self.metric_name = metric_name
        self.vllm_client = VLLMClient(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            tensor_parallel_size=tensor_parallel_size,
            prompt_key=prompt_key,
            additional_keys=additional_keys,
            max_model_len=max_length,
            max_num_seqs=max_num_seqs,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self.scores = None
        self.raw_predictions = None

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
        documents: dict[str, dict[str, str]],
        queries: dict[str, str],
        prompt: str = DEFAULT_FLUENCY_PROMPT,
        batch_size: int = 128,
        num_instances: int = 1,
        shards: int = 12,
        postprocess_regex: str = r"Rating:(.*?)$",  # regex to extract the rating from the raw prediction
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
            rag_answer = predictions[query_id].replace("##Reason:", "Reason:").replace("##Answer:", "Answer:")
            doc_ids = documents[query_id]
            document_text = ""
            for doc_id in doc_ids:
                document_text += f"[{doc_id}]: {documents[query_id][doc_id]}\n"
            query_text = queries[query_id]
            prompt = prompt.replace("{{language}}", ISO_TO_LANG[self.language_code].capitalize())
            prompt = prompt.replace("{{question}}", query_text.strip())
            prompt = prompt.replace("{{documents}}", document_text.strip())
            prompt = prompt.replace("{{rag_answer}}", rag_answer)
            prompts.append(prompt)

        ### Generate the output from the multigpu VLLM client
        outputs = self.vllm_client.batch_call(
            prompts=prompts,
            query_ids=list(documents.keys()),
            batch_size=batch_size,
            num_instances=num_instances,
            shards=shards,
        )

        for output in outputs:
            query_id = output["query_id"]
            raw_prediction = output["output"]
            (rating, _) = self.postprocess(raw_prediction=raw_prediction, regex=postprocess_regex)
            self.scores[query_id[self.metric_name]] = rating
            self.raw_predictions[query_id] = raw_prediction

        ### Logging the average scores
        logger.info("Averaging the scores achieved by the model ...")
        logger.info("-" * 50)
        avg_fluency_score = sum([self.scores[query_id][self.metric_name] for query_id in documents]) / len(documents)
        logger.info(f"Avg Fluency Score: {avg_fluency_score:8.4f}")
        logger.info("-" * 50)

        return self.scores
