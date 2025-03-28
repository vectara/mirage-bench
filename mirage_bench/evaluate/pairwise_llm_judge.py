from __future__ import annotations

import json
import logging
import os
import random

from tqdm import tqdm

from mirage_bench.generate import (
    AnyScaleAPIClient,
    AzureOpenAIClient,
    ClaudeAPIClient,
    CohereAPIClient,
    GeminiAPIClient,
    OpenAIClient,
    VLLMClient,
)

from .util import preprocessing_text

logger = logging.getLogger(__name__)
random.seed(42)

PAIRWISE_PROMPT_DEFAULT = """
Please act as an impartial judge and evaluate the quality of the responses provided \
by two AI assistants tasked to answer the question displayed below, based on a set \
of documents retrieved by a search engine.
You should choose the assistant that best answers the user question based on a set \
of reference documents that may or not be relevant referenced in the IEEE format.\n \
Your evaluation should consider factors such as the correctness, helpfulness, completeness, accuracy, depth, and level of detail of their responses.
Details are only useful if they answer the user question. If an answer \
contains non-relevant details, it should not be preferred over one that only \
use relevant information.
Begin your evaluation by explaining why each answer correctly answers the user \
question. Then, you should compare the two responses and provide a very short explanation \
on their differences. Avoid any position biases and ensure that the order in which \
the responses were presented does not influence your decision. Do not allow the \
length of the responses to influence your evaluation. Be as objective as possible.
After providing your explanation, output your final verdict by strictly following \
this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, \
and "[[C]]" for a tie.

[User Question]
{{query}}

[Reference Documents]
{{documents}}

[The Start of Assistant A's Answer]
{{answer_a}}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{{answer_b}}
[The End of Assistant B's Answer]
"""

CLIENT_MAP = {
    "anyscale": AnyScaleAPIClient,
    "azure_openai": AzureOpenAIClient,
    "claude": ClaudeAPIClient,
    "cohere": CohereAPIClient,
    "gemini": GeminiAPIClient,
    "openai": OpenAIClient,
    "vllm": VLLMClient,
}


class PairwiseLLMJudgeEvaluator:
    def __init__(self, client: str, model_name_or_path: str, wait: int = 60, **kwargs):
        self.client_name = client
        self.judge_name = model_name_or_path
        if client not in CLIENT_MAP:
            raise ValueError(f"Client {client} not supported. Supported clients: {list(CLIENT_MAP.keys())}")
        self.client = CLIENT_MAP[client](model_name_or_path=self.judge_name, wait=wait, **kwargs)
        self.raw_predictions = []

    def postprocess(self, raw_prediction: str, regex: str) -> tuple[str, bool]:
        if "[[A]]" in raw_prediction:
            return ("A", False)
        elif "[[B]]" in raw_prediction:
            return ("B", False)
        elif "[[C]]" in raw_prediction:
            return ("Tie", False)
        else:
            logger.warning(f"either [[A]], [[B]] or [[C]] are not found in answer: {raw_prediction}")
            return (None, True)

    def save(self, output_file: str):
        """Save the scores (and raw_output) to a file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as fout:
            for prediction in self.predictions:
                fout.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    def evaluate(
        self,
        predictions: dict[dict[str, str]],
        all_model_names: list[str],
        documents: dict[str, dict[str, str]],
        queries: dict[str, str],
        prompt: str = PAIRWISE_PROMPT_DEFAULT,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        batch_size: int = 128,
        postprocess_regex: str = r"\[\[(.*?)\]\]",  # regex to extract the rating from the raw prediction
        sample_max: int = 1,
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
        query_ids = random.sample(list(queries.keys()), sample_max)

        for query_id in tqdm(query_ids, desc="Processing queries", total=len(query_ids)):
            logging.info(f"Processing query_id: {query_id}")
            query_text = queries[query_id]
            document_text = ""
            for doc_id in documents[query_id]:
                document_text += f"[{doc_id}]: {documents[query_id][doc_id]}\n"

            ### Randomly shuffle the model names to avoid any bias
            random.shuffle(all_model_names)

            all_prompts, model_combinations = [], []
            for idx in range(len(all_model_names) - 1):
                for idy in range(idx + 1, len(all_model_names)):
                    model_idx, model_idy = all_model_names[idx], all_model_names[idy]
                    # Generate the prompt for the pairwise comparison (randomly shuffle the order of the pair)
                    if random.choice([True, False]):
                        model_idx, model_idy = model_idy, model_idx

                    rag_answer_model_idx = preprocessing_text(predictions[model_idx][query_id], remove_hashtags=False)
                    rag_answer_model_idy = preprocessing_text(predictions[model_idy][query_id], remove_hashtags=False)

                    prompt = prompt.replace("{{query}}", query_text)
                    prompt = prompt.replace("{{documents}}", document_text)
                    prompt = prompt.replace("{{answer_a}}", rag_answer_model_idx)
                    prompt = prompt.replace("{{answer_b}}", rag_answer_model_idy)

                    all_prompts.append(prompt)
                    model_combinations.append((model_idx, model_idy))

            ## Generate the output from the client for the pairwise comparison for a single query_id
            logging.info(f"Generating {len(all_prompts)} pairwise combinations with models: {all_model_names}")
            if self.client_name != "vllm":
                outputs = self.client.batch_call(
                    prompts=all_prompts, batch_size=batch_size, temperature=temperature, max_tokens=max_tokens
                )
            else:
                outputs = self.client.batch_call(
                    prompts=all_prompts,
                    query_ids=model_combinations,
                    batch_size=batch_size,
                    **kwargs,
                )

            for output, (model_idx, model_idy) in zip(outputs, model_combinations):
                verdict, error = self.postprocess(raw_prediction=output, regex=postprocess_regex)
                if not error:
                    self.raw_predictions.append(
                        {
                            "query_id": query_id,
                            "judge": self.judge_name,
                            "model_A": model_idx,
                            "model_B": model_idy,
                            "output": output,
                            "verdict": verdict,
                        }
                    )

        return self.raw_predictions
