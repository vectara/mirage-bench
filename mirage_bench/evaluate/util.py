from __future__ import annotations

import ast
import logging
import re

from .tokenizer import SentenceXTokenizer, StanzaTokenizer

logger = logging.getLogger(__name__)


def replace_non_digits_except_hash(input_string: str) -> str:
    """Replace all non-digit characters except # in a given string."""
    # remove double hashes found in ##Reason or ##Answer
    input_string = input_string.replace("##", "")
    # Define the regex pattern to match all non-digit characters except #
    pattern = r"[^0-9#]"
    # Replace the matched characters with an empty string
    result = re.sub(pattern, "", input_string)
    return result


def preprocessing_text(text: str) -> str:
    # Remove the citations from the text
    return text.strip().split("\n\n")[0].strip()


def find_citations(text: str) -> list[str]:
    # Find all the citations in the text
    return set(re.findall(r"\[[^\]]*\]", text, re.DOTALL))


# Load the predictions and references
def tokenizer_with_citations(
    tokenizer: StanzaTokenizer | SentenceXTokenizer, rag_answer: str, doc_ids: list[str], preprocessing: bool = True
) -> list[dict[str, str | list[str]]]:
    output_sentences = []
    # Extract the context from the RAG answer present before Answer:
    context = re.search(r"Reason(.*?)Answer:", rag_answer, re.DOTALL)

    sentences = []
    if context:
        # get the citations from the context
        context_string = context.group(1)
        if preprocessing:
            context_string = preprocessing_text(context_string)

        # segment the content into sentences
        sentences = tokenizer.segment(context_string)

    for sentence in sentences:
        citation_set = set()
        citations = find_citations(sentence)

        # check if there are any citations in the sentence
        if citations:
            for citation in citations:
                parsed_citation = citation.replace("[", "").replace("]", "")
                # Check if there are multiple citations: [doc_1, doc_2, doc_3]
                if "," in parsed_citation:
                    for cit in parsed_citation.split(","):
                        cit_parsed = replace_non_digits_except_hash(cit)
                        try:
                            if ast.literal_eval(cit_parsed) in doc_ids:
                                sentence = sentence.replace(
                                    citation, ""
                                )  # remove the whole citation from the sentence
                                citation_set.add(ast.literal_eval(cit_parsed))  # add the citation to the set
                        except SyntaxError:
                            logger.warning(
                                f"Failed to parse document ID: {cit} ---> Not in set of available document IDs: {list(doc_ids.keys())}"
                            )
                # single citation: [doc_1]
                else:
                    parsed_citation = replace_non_digits_except_hash(parsed_citation)
                    if parsed_citation in doc_ids:
                        sentence = sentence.replace(citation, "")  # remove the whole citation from the sentence
                        citation_set.add(parsed_citation)  # add the citation to the set
        if citation_set:
            output_sentences.append({"text": sentence, "citations": list(citation_set)})

    return output_sentences
