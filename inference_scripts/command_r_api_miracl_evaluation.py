# Make sure to have the newest Cohere SDK installed:
# pip install -U cohere
# Get your free API key from: www.cohere.com

import cohere
import json, tqdm, csv, time
from nltk.tokenize import sent_tokenize
import datasets
import argparse, os
from tqdm import tqdm
from typing import Dict, Union, List, Optional

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
    parser.add_argument("--temperature", required=False, type=float, default=0.3)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
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
    if args.filter_end is None: args.filter_end = len(hf_dataset)
    # datasets_list = [datasets.Dataset.from_list(load_jsonl_file(i)) for i in .dataset_name]
    # hf_dataset = datasets.concatenate_datasets(datasets_list)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loaded {len(hf_dataset)} prompts for {args.language}...")
    output_filepath = f"{args.filename}-{args.filter_start}-{args.filter_end}.jsonl"
    
    if os.path.exists(os.path.join(args.output_dir, output_filepath)):
        print(f"File {output_filepath} already exists. No need to rerun the experiment.")
        exit(0)

    cohere_key = "XXXX"
    co = cohere.Client(cohere_key)

    if "command-r-plus" in args.model:
        final_model_name = "command-r-plus"
    elif "c4ai-aya-23" in args.model:
        final_model_name = "c4ai-aya-23"
    elif "command-r" in args.model:
        final_model_name = "command-r"

    all_data = []

    for idx, row in enumerate(hf_dataset):
        prompt = row["prompt"]
        query_id = row["query_id"]
        positive_ids = row["positive_ids"]
        negative_ids = row["negative_ids"]
        all_data.append({
            "prompt": prompt,
            "query_id": query_id,
            "positive_ids": positive_ids,
            "negative_ids": negative_ids
        })
    
    all_data = all_data[args.filter_start:args.filter_end]
    final_output = {}
    idx = 0

    for data in tqdm(all_data, total=len(all_data), desc="Generating outputs..."):
        idx += 1
        
        prompt = data["prompt"]
        query_id = data["query_id"]
        positive_ids = data["positive_ids"]
        negative_ids = data["negative_ids"]
        
        try:
            response = co.chat(
                model=final_model_name,
                message=prompt,
                temperature=args.temperature,
            )
            
            output = response.text
            final_output[idx] = {
                "outputs": {f"{args.model}": output},
                "prompt": prompt,
                "query_id": query_id,
                "positive_ids": positive_ids,
                "negative_ids": negative_ids
            }

            save_results(args.output_dir, final_output, output_filepath)
            
        
        except Exception as e:
            print(f"Error for query_id: {query_id}. Error: {e}")
            
            try:
                response = co.chat(
                    model=final_model_name,
                    message=prompt,
                    temperature=args.temperature,
                )
                
                output = response.text
                
                final_output[idx] = {
                    "outputs": {f"{args.model}": output},
                    "prompt": prompt,
                    "query_id": query_id,
                    "positive_ids": positive_ids,
                    "negative_ids": negative_ids
                }

                save_results(args.output_dir, final_output, output_filepath)
            
            except Exception as e:
                print(f"Retry 1 Error for query_id: {query_id}. Error: {e}")

                final_output[idx] = {
                    "outputs": {f"{args.model}": None},
                    "prompt": prompt,
                    "query_id": query_id,
                    "positive_ids": positive_ids,
                    "negative_ids": negative_ids
                }

