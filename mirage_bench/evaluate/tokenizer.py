from __future__ import annotations

import stanza
from sentencex import segment


# Stanza Tokenizer (default) or SentenceX Tokenizer
class StanzaTokenizer:
    def __init__(self, language_code: str):
        self.nlp = stanza.Pipeline(lang=language_code, processors="tokenize")

    def segment(self, text: str) -> list[str]:
        doc = self.nlp(text)
        return [sentence.text for sentence in doc.sentences]


# SentenceX Tokenizer when Stanza is not available
class SentenceXTokenizer:
    def __init__(self, language_code: str):
        self.language_code = language_code

    def segment(self, text):
        return list(segment(self.language_code, text))
