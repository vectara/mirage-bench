from __future__ import annotations

import logging

import evaluate
from multilingual_rouge import rouge_scorer
from tqdm import tqdm

from .util import ISO_TO_LANG, parse_text_wo_citation

logger = logging.getLogger(__name__)

WITH_STEMMER = ["ar", "bn", "hi", "en", "fi", "fr", "de", "ru", "es"]
WITHOUT_STEMMER = ["zh", "th", "ja"]


class RougeBleuEvaluator:
    def __init__(
        self, language_code: str, with_stemmer: list[str] = WITH_STEMMER, answer_regex: str = r"Answer:(.*?)$"
    ):
        self.language_code = language_code
        self.sacrebleu_scorer = evaluate.load("sacrebleu")
        use_stemmer = True if language_code in with_stemmer else False
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=use_stemmer, lang=ISO_TO_LANG[language_code]
        )
        self.scores = None
        self.answer_regex = answer_regex

    def evaluate(
        self,
        predictions: dict[str, str],
        reference_predictions: dict[str, str],
        documents: dict[str, dict[str, str]],
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate the model on the given predictions and return the scores.
        Args:
            predictions (dict): The predictions of the model, where the key is the query_id and the value is the RAG
            do
        """
        self.scores = {query_id: {"answer_bleu": 0, "answer_rougeL": 0} for query_id in documents}

        for query_id in tqdm(documents, desc="Processing queries", total=len(documents)):
            # parse the answer from the RAG answer
            doc_ids = documents[query_id]  # get the doc_ids for the query_id
            rag_answer = parse_text_wo_citation(predictions[query_id], regex=self.answer_regex, doc_ids=doc_ids)
            reference_rag_answer = parse_text_wo_citation(
                reference_predictions[query_id], regex=self.answer_regex, doc_ids=doc_ids
            )

            # compute the BLEU and RougeL scores
            sacrebleu_score = self.sacrebleu_scorer.compute(
                predictions=[rag_answer], references=[reference_rag_answer]
            )
            rouge_score = self.rouge_scorer.score(rag_answer, reference_rag_answer)

            # store the scores
            self.scores[query_id]["answer_bleu"] = round(sacrebleu_score["score"] / 100, 4)
            self.scores[query_id]["answer_rougeL"] = round(rouge_score["rougeL"].fmeasure, 4)

        # compute the average scores
        average_blue_score = sum([self.scores[query_id]["answer_bleu"] for query_id in self.scores]) / len(documents)
        average_rouge_score = sum([self.scores[query_id]["answer_rougeL"] for query_id in self.scores]) / len(
            documents
        )
        logger.info("Averaging the scores achieved by the model ...")
        logger.info("-" * 50)
        logger.info(f"Avg Answer BLEU:    {average_blue_score:8.4f}")
        logger.info(f"Avg Answer RougeL:  {average_rouge_score:8.4f}")
        logger.info("-" * 50)
        return self.scores
