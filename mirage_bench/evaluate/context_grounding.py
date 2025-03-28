from __future__ import annotations

import logging

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .tokenizer import SentenceXTokenizer, StanzaTokenizer
from .util import tokenizer_with_citations

logger = logging.getLogger(__name__)


class XNLIModel(AutoModelForSequenceClassification):
    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True).to(
            self.device
        )
        self.label_names = ["entailment", "neutral", "contradiction"]

    def predict(self, premises: list[str], hypothesis: list[str], batch_size: int) -> dict[str, float]:
        predictions = []
        with torch.no_grad():
            for itr in tqdm(
                range(0, len(premises), batch_size), desc=f"Computing NLI Scores with batch_size = {batch_size}..."
            ):
                inputs = self.tokenizer(
                    premises[itr : itr + batch_size],
                    hypothesis[itr : itr + batch_size],
                    max_length=self.tokenizer.model_max_length - 2,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                prediction = torch.softmax(outputs["logits"], -1).tolist()  # (batch_size, num_labels)
                predictions += [
                    {name: round(float(pred), 1) for pred, name in zip(preds, self.label_names)}
                    for preds in prediction
                ]

            return predictions


class ContextGroundingEvaluator:
    def __init__(
        self, language_code: str, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    ):
        self.language_code = language_code
        self.xnli_model = XNLIModel(model_name)
        self.scores = None
        try:
            logger.info("\tUsing Stanza Tokenizer...")
            self.sentence_tokenizer = StanzaTokenizer(language_code=language_code)
        except Exception:
            print("\tStanza Tokenizer not available, using SentenceX Tokenizer...")
            self.sentence_tokenizer = SentenceXTokenizer(language_code=language_code)

    def evaluate(
        self, predictions: dict[str, str], documents: dict[str, dict[str, str]], batch_size: int = 128, **kwargs
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
        all_sentences = {query_id: {} for query_id in documents}

        for query_id in tqdm(documents, desc="Processing queries", total=len(documents)):
            rag_answer = predictions[query_id]
            references = documents[query_id]
            sentences_with_citations = tokenizer_with_citations(
                self.sentence_tokenizer, rag_answer, doc_ids=references
            )
            all_sentences[query_id] = sentences_with_citations

        # Computing grounding similarity
        premises, hypothesis, total_count = [], [], []

        for query_id in all_sentences:
            count = 0
            for sentence_dict in all_sentences[query_id]:
                sentence, citations = sentence_dict["text"], sentence_dict["citations"]
                # Count the number of citations
                for doc_id in citations:
                    count += 1
                    premises.append(documents[query_id][doc_id])
                    hypothesis.append(sentence)

            # Store the total count of citations + sentences (will be used for averaging)
            total_count.append(count)

        logger.info(f"Evaluating with model: {self.xnli_model}")
        predictions = self.xnli_model.predict(premises, hypothesis, batch_size=batch_size)
        if not self.scores:
            self.scores = {
                query_id: {"support_entailment": 0, "support_contradiction": 0, "support_neutral": 0}
                for query_id in documents
            }

        start_idx = 0
        for query_id, count in zip(all_sentences, total_count):
            if count > 0:
                entailment_score, contradiction_score, neutral_score = 0, 0, 0
                for prediction in predictions[start_idx : start_idx + count]:
                    entailment_score += prediction["support_entailment"]
                    contradiction_score += prediction["support_contradiction"]
                    neutral_score += prediction["support_neutral"]

                self.scores[query_id]["support_entailment"] += round(entailment_score / count, 4)
                self.scores[query_id]["support_contradiction"] += round(contradiction_score / count, 4)
                self.scores[query_id]["support_neutral"] += round(neutral_score / count, 4)
                start_idx += count

        # Logging the average scores
        logger.info("Averaging the scores achieved by the model ...")
        avg_entailment_score = sum([self.scores[query_id]["support_entailment"] for query_id in documents]) / len(
            documents
        )
        avg_contradiction_score = sum(
            [self.scores[query_id]["support_contradiction"] for query_id in documents]
        ) / len(documents)
        avg_neutral_score = sum([self.scores[query_id]["support_neutral"] for query_id in documents]) / len(documents)
        logger.info("-" * 50)
        logger.info(f"Avg Entailment:    {avg_entailment_score:8.4f}")
        logger.info(f"Avg Neutral:       {avg_neutral_score:8.4f}")
        logger.info(f"Avg Contradiction: {avg_contradiction_score:8.4f}")
        logger.info("-" * 50)
        return self.scores
