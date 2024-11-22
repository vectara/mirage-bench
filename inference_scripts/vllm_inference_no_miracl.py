"""
This example shows how to use Ray Data for running offline batch inference
distributively on a multi-nodes cluster.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""

from vllm import LLM, SamplingParams
from typing import Dict
from transformers import AutoTokenizer
import numpy as np
import ray
import datasets
import json
import os
import argparse
from typing import List, Union, Optional

def load_jsonl_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as fin:
        return [json.loads(row) for row in fin]


def save_results(output_dir: str,
                 results: Dict[str, Dict[str, Union[str, List[str]]]],
                 filename: Optional[str] = 'results.jsonl'):
    """
    Save the results of generated output (results[model_name] ...) in JSONL format.

    Args:
        output_dir (str): The directory where the JSONL file will be saved.
        results (Dict[str, List[str]]): A dictionary containing the model results.
        filename (Optional[str], optional): The name of the JSONL file. Defaults to 'results.jsonl'.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    # Save results in JSONL format
    with open(os.path.join(output_dir, filename), 'w') as f:        
        for idx in results:
            f.write(json.dumps(results[idx], ensure_ascii=False) + '\n')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=None)
    parser.add_argument("--prompt_template", default=None)
    parser.add_argument("--temperature", required=False, type=float, default=0.3)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--max_new_tokens", required=False, type=int, default=4096)
    parser.add_argument("--max_model_len", required=False, type=int, default=4096)
    parser.add_argument("--top_p", required=False, type=float, default=0.95)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--filename", default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)

    args = parser.parse_args()

    terminators, stop_strings = [], []
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if "llama-3" in args.model.lower():
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        stop_strings = ["<|eot_id|>"]

    sampling_params = SamplingParams(temperature=args.temperature,
                                    max_tokens=args.max_new_tokens,
                                    stop_token_ids=terminators if terminators else None,
                                    stop=stop_strings if stop_strings else None, 
                                    )

    # Create a class to do batch inference.
    class LLMPredictor:

        def __init__(self):
            # Create an LLM.
            self.llm = LLM(model=args.model, 
                        max_model_len=args.max_model_len, 
                        max_num_seqs=1, 
                        max_seq_len_to_capture=args.max_model_len,
                        download_dir=args.cache_dir,
                        dtype="bfloat16", 
                        trust_remote_code=True)  # skip graph capturing for faster cold starts)

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            # Generate texts from the prompts.
            # The output is a list of RequestOutput objects that contain the prompt,
            # generated text, and other information.
            outputs = self.llm.generate(batch["prompt_mistral"], sampling_params)
            prompt = []
            generated_text = []
            for output in outputs:
                prompt.append(output.prompt)
                generated_text.append(' '.join([o.text for o in output.outputs]))
            
            return {
                "prompt": batch["prompt"],
                "output": generated_text,
                "query_id": batch["query_id"], 
                "doc_ids": batch["doc_ids"],
                "positive_ids": batch["positive_ids"],
                "negative_ids": batch["negative_ids"]
            }


    # Read one text file from S3. Ray Data supports reading multiple files
    # from cloud storage (such as JSONL, Parquet, CSV, binary format).
    # ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
    hf_dataset = datasets.load_dataset(args.dataset_name, args.language, split=args.split, cache_dir=args.cache_dir)
    # datasets_list = [datasets.Dataset.from_list(load_jsonl_file(i)) for i in .dataset_name]
    # hf_dataset = datasets.concatenate_datasets(datasets_list)
    
    print(f"Loaded {len(hf_dataset)} prompts for {args.language}...")
    
    prompts = []

    for idx, row in enumerate(hf_dataset):
        prompt = row["prompt"]
        query_id = row["query_id"]
        doc_ids = row["doc_ids"]
        positive_ids = row["positive_ids"]
        negative_ids = row["negative_ids"]
        messages = [
        {"role": "user", "content": f"{prompt}"}]
        prompt_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_template)

    hf_dataset = hf_dataset.add_column("prompt_mistral", prompts)

    # Convert the Huggingface dataset to Ray Data.
    ds = ray.data.from_huggingface(hf_dataset)

    # Apply batch inference for all input data.
    ds = ds.repartition(12, shuffle=False)

    ds = ds.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=args.concurrency,
        # Specify the number of GPUs required per LLM instance.
        # NOTE: Do NOT set `num_gpus` when using vLLM with tensor-parallelism
        # (i.e., `tensor_parallel_size`).
        num_gpus=args.num_gpus,
        # Specify the batch size for inference.
        batch_size=args.batch_size,
        zero_copy_batch=True,
    )

    # Peek first 10 results.
    # NOTE: This is for local testing and debugging. For production use case,
    # one should write full result out as shown below.
    outputs = ds.take_all()

    output_dict = {}

    for idx, output in enumerate(outputs):
        generated_text = output["output"]
        prompt = output["prompt"]
        output_dict[idx] = {"output": generated_text, "prompt": prompt}
        output_dict[idx].update({"query_id": output["query_id"], "doc_ids": list(output["doc_ids"]), "positive_ids": list(output["positive_ids"]), "negative_ids": list(output["negative_ids"])})

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the results in JSONL format.
    save_results(args.output_dir, output_dict, f"{args.filename}.jsonl")