from __future__ import annotations

import logging

from FlagEmbedding import FlagReranker as Reranker
from tqdm import tqdm

from .util import filter_citations

logger = logging.getLogger(__name__)


class FlagReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", cache_dir: str = None, use_fp16: bool = True):
        self.reranker = Reranker(model_name, use_fp16=use_fp16, cache_dir=cache_dir)

    def compute_score(self, pairs: list[list[str]], batch_size: int) -> list[float]:
        predictions = []
        for itr in tqdm(
            range(0, len(pairs), batch_size), desc=f"Computing Reranker scores with batch_size = {batch_size}..."
        ):
            end_pointer = len(pairs) if itr + batch_size > len(pairs) else itr + batch_size
            scores = self.reranker.compute_score(pairs[itr:end_pointer])
            if not isinstance(scores, list):
                scores = [scores]
            predictions += scores
        return predictions


class RerankerScoreEvaluator:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", cache_dir: str = None, use_fp16: bool = True):
        self.reranker = FlagReranker(model_name, cache_dir=cache_dir, use_fp16=use_fp16)
        self.scores = None

    def evaluate(
        self,
        predictions: dict[str, str],
        documents: dict[str, dict[str, str]],
        queries: dict[str, str],
        batch_size: int = 128,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        self.scores = {query_id: {"reranker_score": 0} for query_id in documents}

        pairs, counts = [], []
        for query_id in tqdm(documents, desc="Processing queries", total=len(documents)):
            # parse the answer from the RAG answer
            doc_ids = documents[query_id]
            results = filter_citations(predictions[query_id], doc_ids=doc_ids)
            counts.append(len(results["citations"]))
            for doc_id in results["citations"]:
                pairs.append([queries[query_id], documents[query_id][doc_id]])

        #### Compute the reranker scores
        reranker_predictions = self.reranker.compute_score(pairs, batch_size=batch_size)

        start_idx = 0
        for query_id, count in zip(documents, counts):
            if count > 0:
                reranker_score = 0
                for score in reranker_predictions[start_idx : start_idx + count]:
                    reranker_score += score
                self.scores[query_id]["reranker_score"] = reranker_score / count
                start_idx += count

        #### logging the average reranker score
        avg_reranker_score = sum([self.scores[query_id]["reranker_score"] for query_id in documents]) / len(documents)
        logging.info("Averaging the scores achieved by the model ...")
        logging.info("-" * 50)
        logging.info(f"Avg Reranker Score: {avg_reranker_score:8.4f}")
        logging.info("-" * 50)
        return self.scores
