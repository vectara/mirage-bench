from __future__ import annotations

import logging

import pytrec_eval
from tqdm import tqdm

from .util import citations_in_order

logger = logging.getLogger(__name__)


class ContextMAPRecallEvaluator:
    def __init__(self, language_code: str, k_values: list[int] = [10]):
        self.language_code = language_code
        self.k_values = k_values
        self.map_string = "map_cut." + ",".join([str(k) for k in self.k_values])
        self.recall_string = "recall." + ",".join([str(k) for k in self.k_values])
        self.scores = None

    def evaluate(
        self,
        predictions: dict[str, str],
        documents: dict[str, dict[str, str]],
        qrels: dict[str, dict[str, int]],
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate the model on the given predictions and return the scores.
        Args:
            predictions (dict): The predictions of the model, where the key is the query_id and the value is the RAG answer.
            documents (dict): The document_id and document text for each query_id.
        Returns:
            dict: The entailment, contrdiction and neutral scores for each query_id.
        """
        self.scores, runfile = {}, {}
        for query_id in documents:
            self.scores[query_id] = {f"citation_MAP@{k_value}": 0 for k_value in self.k_values}
            self.scores[query_id].update({f"citation_Recall@{k_value}": 0 for k_value in self.k_values})

        for query_id in tqdm(documents, desc="Processing queries", total=len(documents)):
            rag_answer = predictions[query_id]
            doc_ids = documents[query_id]
            runfile[query_id] = citations_in_order(rag_answer, doc_ids)

        logging.info("Evaluating with pytrec eval ...")
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {self.map_string, self.recall_string})
        scores = evaluator.evaluate(runfile)

        for query_id in scores:
            for k in self.k_values:
                self.scores[query_id][f"citation_MAP@{k}"] = scores[query_id]["map_cut_" + str(k)]
                self.scores[query_id][f"citation_Recall@{k}"] = scores[query_id]["recall_" + str(k)]

        #### Logging the average scores
        logger.info("Averaging the scores achieved by the model ...")
        for k in self.k_values:
            avg_map_score = sum([self.scores[query_id][f"citation_MAP@{k}"] for query_id in documents]) / len(
                documents
            )
            avg_recall_score = sum([self.scores[query_id][f"citation_Recall@{k}"] for query_id in documents]) / len(
                documents
            )
            logger.info("-" * 50)
            logger.info(f"Avg MAP@{k}:    {avg_map_score:8.4f}")
            logger.info(f"Avg Recall@{k}: {avg_recall_score:8.4f}")
            logger.info("-" * 50)

        return self.scores
