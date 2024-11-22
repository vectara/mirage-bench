# Make sure to have the newest Cohere SDK installed:
# pip install -U cohere
# Get your free API key from: www.cohere.com

import requests
import json, tqdm, csv, time
from nltk.tokenize import sent_tokenize
import datasets
import argparse, os
from tqdm import tqdm
from typing import Dict, Union, List, Optional

ISO_MAP = {
    "ar": "Arabic",
    "bn": "Bengali",
    "fi": "Finnish",
    "id": "Indonesian",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "sw": "Swahili",
    "te": "Telugu",
    "th": "Thai",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "hi": "Hindi",
    "zh": "Chinese",
    "en": "English",
    "yo": "Yoruba",
    "de": "German"
    }

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
    parser.add_argument("--filter_start", type=int, default=None)
    parser.add_argument("--filter_end", type=int, default=None)

    args = parser.parse_args()

    # Read one text file from S3. Ray Data supports reading multiple files
    # from cloud storage (such as JSONL, Parquet, CSV, binary format).
    # ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
    hf_dataset = datasets.load_dataset(args.dataset_name, args.language, split=args.split, cache_dir=args.cache_dir)
    # datasets_list = [datasets.Dataset.from_list(load_jsonl_file(i)) for i in .dataset_name]
    # hf_dataset = datasets.concatenate_datasets(datasets_list)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loaded {len(hf_dataset)} prompts for {args.language}...")
    output_filepath = f"{args.filename}-{args.filter_start}-{args.filter_end}.jsonl"
    
    if os.path.exists(os.path.join(args.output_dir, output_filepath)):
        print(f"File {output_filepath} already exists. No need to rerun the experiment.")
        exit(0)


    api_base = os.getenv("ANYSCALE_BASE_URL")
    token = os.getenv("ANYSCALE_API_KEY")
    url = f"{api_base}/chat/completions"
    s = requests.Session()

    all_data = []

    for idx, row in enumerate(hf_dataset):
        prompt = row["prompt"]
        query_id = row["query_id"]
        doc_ids = row["doc_ids"]
        documents = row["documents"]
        gold_answers = row["gold_answer"]
        
        all_data.append({
            "prompt": prompt,
            "query_id": query_id,
            "doc_ids": doc_ids,
            "documents": documents,
            "gold_answers": gold_answers
        })
    
    if args.filter_start > len(all_data):
        print("Filter start or end index is out of range. Exiting...")
        exit(0)
    elif args.filter_end > len(all_data):
        args.filter_end = len(all_data)
    
    all_data = all_data[args.filter_start:args.filter_end]
    
    final_output = {}
    idx = 0

    for data in tqdm(all_data, total=len(all_data), desc="Generating outputs..."):
        idx += 1
        
        prompt = data["prompt"]
        query_id = data["query_id"]
        
        try:
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": f"{prompt}"}],
                "temperature": args.temperature
            }

            with s.post(url, headers={"Authorization": f"Bearer {token}"}, json=body) as resp:
                output = resp.json()["choices"][0]["message"]["content"]
                            
                final_output[idx] = {
                    "outputs": {f"{args.model}": output},
                    "prompt": prompt,
                    "query_id": query_id,
                    "doc_ids": data["doc_ids"],
                    "documents": data["documents"],
                    "gold_answers": data["gold_answers"]
                }

                if idx > 10 and idx % 10 == 0:
                    save_results(args.output_dir, final_output, output_filepath)
            
        
        except Exception as e:
            print(f"Error for query_id: {query_id}. Error: {e}")
            time.sleep(60)
            
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": f"{prompt}"}],
                "temperature": args.temperature
            }

            with s.post(url, headers={"Authorization": f"Bearer {token}"}, json=body) as resp:
                output = resp.json()["choices"][0]["message"]["content"]
                            
                final_output[idx] = {
                    "outputs": {f"{args.model}": output},
                    "prompt": prompt,
                    "query_id": query_id,
                    "doc_ids": data["doc_ids"],
                    "documents": data["documents"],
                    "gold_answers": data["gold_answers"]
                }

                save_results(args.output_dir, final_output, output_filepath)
    

    # Save the final results
    save_results(args.output_dir, final_output, output_filepath)