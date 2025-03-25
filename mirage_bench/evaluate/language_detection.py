from __future__ import annotations

import logging

from langdetect import detect_langs
from lingua import Language, LanguageDetectorBuilder
from tqdm import tqdm

from .util import parse_text_wo_citation

logger = logging.getLogger(__name__)

LANG_TO_CODE = {
    "YORUBA": "yo",
    "ENGLISH": "en",
}


# langdetect for all languages except yoruba (yo)
# https://github.com/Mimino666/langdetect
class LangDetect:
    def __init__(self):
        pass

    def detect(self, text: str) -> dict[str, float]:
        lang_detect = {}
        for lang_object in detect_langs(text):
            if lang_object.lang == "zh-cn":
                lang_detect["zh"] = lang_object.prob
            else:
                lang_detect[lang_object.lang] = lang_object.prob
        return lang_detect


# for yoruba language (using Lingua-py)
# https://github.com/pemistahl/lingua-py
class YorubaDetect:
    def __init__(self):
        self.languages = [Language.ENGLISH, Language.YORUBA]
        self.yoruba_detector = LanguageDetectorBuilder.from_languages(*self.languages).build()

    def detect(self, text: str) -> dict[str, float]:
        lang_detect = {}
        confidence_values = self.yoruba_detector.compute_language_confidence_values(text)
        for confidence in confidence_values:
            lang_detect[LANG_TO_CODE[confidence.language.name]] = confidence.value
        return lang_detect


class LanguageDetectionEvaluator:
    def __init__(self, language_code: str):
        self.language_code = language_code
        if language_code == "yo":
            self.detector = YorubaDetect()
        else:
            self.detector = LangDetect()
        self.results = None

    def evaluate(
        self, predictions: dict[str, str], documents: dict[str, dict[str, str]], **kwargs
    ) -> dict[str, dict[str, float]]:
        self.scores = {query_id: {"target_language": 0, "english": 0, "other": 0} for query_id in documents}

        for query_id in tqdm(documents, desc="Processing queries", total=len(documents)):
            rag_output = predictions[query_id]
            parsed_text = parse_text_wo_citation(rag_output, regex=None, doc_ids=documents[query_id])
            detected_langs = self.detector.detect(parsed_text)
            # store the scores
            for detected_lang in detected_langs:
                if detected_lang == self.language_code:
                    self.scores[query_id]["target_language"] = detected_langs[self.language_code]
                if detected_lang == "en":
                    self.scores[query_id]["english_language"] = detected_langs["en"]
                else:
                    self.scores[query_id]["other_language"] += detected_langs[detected_lang]

        # compute the average scores
        avg_target_lang = sum([self.scores[query_id]["target_language"] for query_id in self.scores]) / len(documents)
        avg_english = sum([self.scores[query_id]["english_language"] for query_id in self.scores]) / len(documents)
        avg_other = sum([self.scores[query_id]["other_language"] for query_id in self.scores]) / len(documents)

        logger.info("Averaging the scores achieved by the model ...")
        logger.info("-" * 50)
        logger.info(f"Avg Target Language: {avg_target_lang:8.4f}")
        logger.info(f"Avg English:         {avg_english:8.4f}")
        logger.info(f"Avg Other:           {avg_other:8.4f}")
        logger.info("-" * 50)
        return self.scores
