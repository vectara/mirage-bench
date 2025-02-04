from __future__ import annotations

import json
import os


def load_jsonl(input_filepath: str) -> list[dict[str, str | list[str]]]:
    """
    Load a JSONL file and return its contents as a list.
    """
    if not input_filepath.endswith(".jsonl"):
        raise ValueError("The input file must be a valid JSONL file.")

    with open(input_filepath, encoding="utf-8") as fin:
        return [json.loads(row) for row in fin]

def save_jsonl(output_filepath: str,
               results: dict[str, dict[str, str | list[str]]]) -> None:
    """
    Save the results of generated output (results[model_name] ...) in JSONL format.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Save results in JSONL format
    with open(output_filepath, 'w') as fout:
        for idx in results:
            fout.write(json.dumps(results[idx], ensure_ascii=False) + '\n')
