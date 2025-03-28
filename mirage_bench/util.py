from __future__ import annotations

import json
import os

import datasets
from tqdm.auto import tqdm


def load_prompts(dataset_name: str, language_code: str, split: str = "dev") -> dict[str, str]:
    prompts = {}
    hf_dataset = datasets.load_dataset(dataset_name, language_code, split=split)

    for row in tqdm(hf_dataset, total=len(hf_dataset), desc="Loading prompts"):
        prompts[row["query_id"]] = row["prompt"]
    return prompts


def load_queries(dataset_name: str, language_code: str, split: str = "dev") -> dict[str, str]:
    queries = {}
    hf_dataset = datasets.load_dataset(dataset_name, language_code, split=split)

    for row in tqdm(hf_dataset, total=len(hf_dataset), desc="Loading queries"):
        queries[row["query_id"]] = row["prompt"].split("Question:")[1].split("\n\nContexts:")[0].strip()

    return queries


def load_qrels(dataset_name: str, language_code: str, split: str = "dev") -> dict[str, dict[str, int]]:
    qrels = {}
    hf_dataset = datasets.load_dataset(dataset_name, language_code, split=split)

    for row in tqdm(hf_dataset, total=len(hf_dataset), desc="Loading qrels"):
        query_id = row["query_id"]
        qrels[query_id] = {doc_id: 1 for doc_id in row["positive_ids"]}
        qrels[query_id].update({doc_id: 0 for doc_id in row["negative_ids"]})

    return qrels


def load_documents(dataset_name: str, language_code: str, split: str = "dev") -> dict[str, str]:
    documents_dict = {}
    hf_dataset = datasets.load_dataset(dataset_name, language_code, split=split)

    for row in tqdm(hf_dataset, total=len(hf_dataset), desc="Loading documents"):
        query_id = row["query_id"]
        context = row["prompt"].split("\n\nContexts:")[1].split("\n\nInstruction")[0].strip()

        # Get the positive and negative document ids
        doc_ids = row["positive_ids"] + row["negative_ids"]

        start_ids = [context.find(f"[{doc_id}]") for doc_id in doc_ids]
        sorted_doc_ids = [x for _, x in sorted(zip(start_ids, doc_ids))]

        documents_dict[query_id] = {}
        # start from the first document until the second last one
        # Take the text between the two document ids: [doc_id] ..... [next_doc_id]
        for idx in range(len(sorted_doc_ids[:-1])):
            doc_id, next_doc_id = sorted_doc_ids[idx], sorted_doc_ids[idx + 1]
            doc_text = context.split(f"[{doc_id}]")[1].split(f"[{next_doc_id}]")[0].strip()
            documents_dict[query_id][doc_id] = doc_text

        # last doc id
        doc_id = sorted_doc_ids[-1]
        doc_text = context.split(f"[{doc_id}]")[1].strip()
        documents_dict[query_id][doc_id] = doc_text

    return documents_dict


def load_predictions(dataset_name: str, model_name: str, language_code: str, split: str = "dev") -> dict[str, str]:
    predictions = {}
    hf_dataset = datasets.load_dataset(dataset_name, language_code, split=split)

    for row in tqdm(hf_dataset, total=len(hf_dataset), desc="Loading predictions"):
        query_id = row["query_id"]
        for output_row in row["outputs"]:
            if output_row["model"] == model_name:
                predictions[query_id] = output_row["output"]
                break
    return predictions


def save_results(
    output_dir: str, results: dict[str, dict[str, str | list[str]]], filename: str | None = "results.jsonl"
):
    """
    Save the results of generated output (results[model_name] ...) in JSONL format.

    Args:
        output_dir: The directory where the JSONL file will be saved.
        results: A dictionary containing the model results.
        filename: The name of the JSONL file. Defaults to 'results.jsonl'.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Save results in JSONL format
    with open(os.path.join(output_dir, filename), "w") as f:
        for idx in results:
            f.write(json.dumps(results[idx], ensure_ascii=False) + "\n")


def load_results(input_filepath: str) -> list[dict[str, str | list[str]]]:
    """
    Load a JSONL file and return its contents as a list.
    """
    if not input_filepath.endswith(".jsonl"):
        raise ValueError("The input file must be a valid JSONL file.")

    with open(input_filepath, encoding="utf-8") as fin:
        return [json.loads(row) for row in fin]
