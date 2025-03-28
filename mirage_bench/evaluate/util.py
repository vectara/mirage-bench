from __future__ import annotations

import ast
import logging
import re

from .tokenizer import SentenceXTokenizer, StanzaTokenizer

logger = logging.getLogger(__name__)

ISO_TO_LANG = {
    "ar": "arabic",
    "bn": "bengali",
    "hi": "hindi",
    "en": "english",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "ru": "russian",
    "es": "spanish",
    "zh": "chinese",
    "th": "thai",
    "ja": "japanese",
    "sw": "swahili",
    "yo": "yoruba",
    "fa": "persian",
    "id": "indonesian",
    "ko": "korean",
    "te": "telugu",
}


##### EASY UTIL FUNCTIONS #####
def preprocessing_text(text: str, remove_hashtags: bool = True) -> str:
    # Remove the citations from the text
    if remove_hashtags:
        return text.strip().split("\n\n")[0].strip().replace("##Reason:", "").replace("##Answer:", "").strip()
    else:
        return (
            text.strip()
            .split("\n\n")[0]
            .strip()
            .replace("##Reason:", "Reason:")
            .replace("##Answer:", "Answer:")
            .strip()
        )


def parse(text: str, regex: str = None) -> str:
    """Generic function to parse the text using the given regex."""
    parsed_string = ""
    if regex:
        text_group = re.search(regex, text, re.DOTALL)
        if text_group:
            parsed_string = preprocessing_text(text_group.group(1))
    else:
        parsed_string = preprocessing_text(text)
    return parsed_string


def find_all_citations(text: str) -> list[str]:
    # Find all the citations in the text
    return set(re.findall(r"\[[^\]]*\]", text, re.DOTALL))


def replace_non_digits_except_hash(input_string: str) -> str:
    """Replace all non-digit characters except # in a given string."""
    # remove double hashes found in ##Reason or ##Answer
    input_string = input_string.replace("##", "")
    # Define the regex pattern to match all non-digit characters except #
    pattern = r"[^0-9#]"
    # Replace the matched characters with an empty string
    result = re.sub(pattern, "", input_string)
    return result


def filter_citations(text: str, doc_ids: list[str]) -> dict[str, str | list[str]]:
    citation_set, citations = [], []
    citations = find_all_citations(text)

    for citation in citations:
        # remove the start and end brackets from the citation
        citation_text = citation.replace("[", "").replace("]", "")
        # Check if there are multiple citations using comma, e.g., doc_1, doc_2, doc_3
        if "," in citation_text:
            for ind_citation in citation_text.split(","):
                ind_citation = replace_non_digits_except_hash(ind_citation)
                try:
                    if ast.literal_eval(ind_citation) in doc_ids:
                        text = text.replace(citation, "")  # remove the original citation from the sentence
                        citation_set.append(ast.literal_eval(ind_citation))  # add the citation to the set
                except SyntaxError:
                    logger.warning(
                        f"Failed to parse document ID: {ind_citation} in {citation} ---> Not in set of document IDs: {list(doc_ids.keys())}"
                    )
        # single citation, e.g., doc_1
        else:
            parsed_citation = replace_non_digits_except_hash(citation_text)
            if citation_text in doc_ids:
                text = text.replace(citation, "")  # remove the original citation from the sentence
                citation_set.append(parsed_citation)  # add the citation to the set

    return {"text": text, "citations": citation_set}


##### SPECIFIC UTIL FUNCTIONS #####


# Used in rouge_and_blue.py
def parse_text_wo_citation(text: str, regex: str, doc_ids: list[str]) -> dict[str, str]:
    """Parse the string and remove the citations from the provided text."""
    text = parse(text=text, regex=regex)
    filtered_output = filter_citations(text, doc_ids)
    return filtered_output["text"]


# Used in context_map_recall.py
def citations_in_order(rag_answer: str, doc_ids: list[str]) -> list[str]:
    runfile = {}

    filtered_output = filter_citations(text=rag_answer, doc_ids=doc_ids)
    citations = filtered_output.get("citations")

    if citations:
        for idx, citation in enumerate(list(dict.fromkeys(citations))):
            runfile[citation] = len(citations) - idx
    return runfile


# Used in context_grounding.py
def tokenizer_with_citations(
    tokenizer: StanzaTokenizer | SentenceXTokenizer, rag_answer: str, doc_ids: list[str]
) -> list[dict[str, str | list[str]]]:
    output_sentences = []
    # Extract the context from the RAG answer present before Answer:
    context = re.search(r"Reason(.*?)Answer:", rag_answer, re.DOTALL)

    sentences = []
    if context:
        # get the citations from the context
        context_string = context.group(1)
        context_string = preprocessing_text(context_string)

        # segment the content into sentences
        sentences = tokenizer.segment(context_string)

    for sentence in sentences:
        filtered_output = filter_citations(sentence, doc_ids)
        citation_set = list(set(filtered_output.get("citations")))
        text_without_citations = filtered_output.get("text")
        if citation_set:
            output_sentences.append({"text": text_without_citations, "citations": citation_set})

    return output_sentences
