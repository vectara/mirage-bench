"""
This is to run Gemma2 models using huggingface inference
"""

from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import ray
import datasets
import json
import os
import argparse
import torch
from tqdm import tqdm
from typing import List, Union, Optional

MAX_LENGTH = 8192

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


class Gemma2Model:
    def __init__(self, weights_path, cache_dir=None):
        self.torch_dtype = torch.bfloat16
        self.weights_path = weights_path
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        assert (
            self.weights_path is not None
        ), "A path to model weights must be defined for using this generator."
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.weights_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.weights_path, 
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
    
    def batch_call(self, prompts, batch_size=1, **kwargs):
        batches = [
            prompts[i : i + batch_size]
            for i in range(0, len(prompts), batch_size)
        ]

        results = []
        for i, batch in enumerate(
            tqdm(batches, desc="Collecting responses", leave=False)
        ):
            responses = self.__call__(batch, **kwargs)
            results.extend(responses)
        
        return results

    def __call__(self, prompts, top_p, temperature, max_new_tokens, min_new_tokens, **kwargs):
        
        responses = []
        
        for prompt in prompts:
            chat = [
                {"role": "user", "content": f"{prompt}"},
            ]
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)        
            inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            outputs = self.model.generate(input_ids=inputs.to(self.model.device),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens)

            response = self.tokenizer.decode(outputs[0])
            responses.append(response)

        return responses
    
    def truncate_response(self, response, max_length=500):
        tokens = self.tokenizer.tokenize(response)[:max_length]
        return self.tokenizer.convert_tokens_to_string(tokens)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=None)
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
    parser.add_argument("--filter_start", type=int, default=0)
    parser.add_argument("--filter_end", type=int, default=None)

    args = parser.parse_args()

    # Read one text file from S3. Ray Data supports reading multiple files
    # from cloud storage (such as JSONL, Parquet, CSV, binary format).
    # ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
    hf_dataset = datasets.load_dataset(args.dataset_name, args.language, split=args.split, cache_dir=args.cache_dir)
    # datasets_list = [datasets.Dataset.from_list(load_jsonl_file(i)) for i in .dataset_name]
    # hf_dataset = datasets.concatenate_datasets(datasets_list)

    if args.filter_end is None: args.filter_end = len(hf_dataset)
    
    print(f"Loaded {len(hf_dataset)} prompts for {args.language}...")
    output_filepath = f"{args.filename}-{args.filter_start}-{args.filter_end}.jsonl"
    
    if os.path.exists(os.path.join(args.output_dir, output_filepath)):
        print(f"File {output_filepath} already exists. No need to rerun the experiment.")
        exit(0)

    prompts = []

    for idx, row in enumerate(hf_dataset):
        prompt = row["prompt"]
        query_id = row["query_id"]
        positive_ids = row["positive_ids"]
        negative_ids = row["negative_ids"]
        prompts.append(prompt)

    gemma_model = Gemma2Model(args.model, cache_dir=args.cache_dir)

    for i in range(args.filter_start, args.filter_end, args.batch_size):
        batch = prompts[i:i+args.batch_size]
        
        generated_text = gemma_model.batch_call(batch, 
                                                batch_size=args.batch_size,
                                                top_p=args.top_p, 
                                                temperature=args.temperature, 
                                                max_new_tokens=args.max_new_tokens, 
                                                min_new_tokens=args.max_new_tokens)
        print(generated_text)
        break
    
    
    # hf_dataset = hf_dataset.add_column("prompt_mistral", prompts)

    # if args.filter_start > len(prompts):
    #     print(f"Filter start is greater than the number of prompts. Exiting...")
    #     exit(0)
    # elif args.filter_end > len(prompts):
    #     args.filter_end = len(prompts)
    
    # hf_dataset = hf_dataset.select(range(args.filter_start, args.filter_end))

    # # Convert the Huggingface dataset to Ray Data.
    # ds = ray.data.from_huggingface(hf_dataset)

    # # Apply batch inference for all input data.
    # ds = ds.repartition(12, shuffle=False)

    # ds = ds.map_batches(
    #     LLMPredictor,
    #     # Set the concurrency to the number of LLM instances.
    #     concurrency=args.concurrency,
    #     # Specify the number of GPUs required per LLM instance.
    #     # NOTE: Do NOT set `num_gpus` when using vLLM with tensor-parallelism
    #     # (i.e., `tensor_parallel_size`).
    #     num_gpus=args.num_gpus,
    #     # Specify the batch size for inference.
    #     batch_size=args.batch_size,
    #     zero_copy_batch=True,
    # )

    # # Peek first 10 results.
    # # NOTE: This is for local testing and debugging. For production use case,
    # # one should write full result out as shown below.
    # outputs = ds.take_all()

    # output_dict = {}

    # for idx, output in enumerate(outputs):
    #     generated_text = output["output"]
    #     output_dict[idx] = {"outputs": {args.model: generated_text}, "prompt": output["prompt"]}
    #     output_dict[idx].update({"query_id": output["query_id"],
    #                               "positive_ids": list(output["positive_ids"]), 
    #                               "negative_ids": list(output["negative_ids"])})

    # os.makedirs(args.output_dir, exist_ok=True)
    
    # # Save the results in JSONL format.
    # save_results(args.output_dir, output_dict, f"{args.filename}-{args.filter_start}-{args.filter_end}.jsonl")