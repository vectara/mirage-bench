"""
This example shows how to use Ray Data for running offline batch inference
distributively on a multi-nodes cluster.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""

from vllm import LLM, SamplingParams
from typing import Dict
import numpy as np
import ray
import datasets
import json
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional

from src.prompts.utils import load_prompt_template


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
    parser.add_argument("--filter_start", type=int, default=0)
    parser.add_argument("--filter_end", type=int, default=2500)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_filepath = f"{args.filename}-final-{args.filter_start}-{args.filter_end}.jsonl"
    
    if os.path.exists(os.path.join(args.output_dir, output_filepath)):
        print(f"File {output_filepath} already exists. No need to rerun the experiment.")
    else:
        temp_model = AutoModel.from_pretrained(args.model, cache_dir=args.cache_dir)
        del temp_model
        
        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=args.temperature,
                                        top_p=args.top_p, 
                                        max_tokens=args.max_new_tokens)
        prompt_cls = load_prompt_template(args.prompt_template)


        # Create a class to do batch inference.
        class LLMPredictor:

            def __init__(self):
                # Create an LLM.
                self.llm = LLM(model=args.model, max_model_len=args.max_model_len, max_num_seqs=1, 
                               download_dir=args.cache_dir)

            def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
                # Generate texts from the prompts.
                # The output is a list of RequestOutput objects that contain the prompt,
                # generated text, and other information.
                outputs = self.llm.generate(batch["translation_prompt"], sampling_params)
                prompt = []
                generated_text = []
                for output in outputs:
                    prompt.append(output.prompt)
                    generated_text.append(' '.join([o.text for o in output.outputs]))
                
                return {
                    "translation_prompt": prompt,
                    "generated_text": generated_text,
                    "id": batch["id"],
                    "en_chosen": batch["en_chosen"],
                    "en_rejected": batch["en_rejected"],
                    "en_input": batch["en_input"],
                    "selection": batch["selection"],
                    "source": batch["source"],
                }


        # Read one text file from S3. Ray Data supports reading multiple files
        # from cloud storage (such as JSONL, Parquet, CSV, binary format).
        # ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
        hf_dataset = datasets.load_dataset(args.dataset_name, split=args.split, cache_dir=args.cache_dir)

        source_list, row_ids, turn_ids, translation_prompts, inputs_list, chosen_list, rejected_list = [], [], [], [], [], [], []

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        for idx, row in enumerate(hf_dataset):
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]
            source = row["source"]

            for response in chosen:
                if response["role"] == "assistant":
                    chosen_string = response["content"]
            
            for response in rejected:
                if response["role"] == "assistant":
                    rejected_string = response["content"]

            positive_prompt = prompt_cls(question=prompt, chosen=chosen_string, language=args.language)
            negative_prompt = prompt_cls(question=prompt, chosen=rejected_string, language=args.language)
            tok_positive_prompt, tok_negative_prompt = len(tokenizer.tokenize(positive_prompt)), len(tokenizer.tokenize(negative_prompt))

            if tok_positive_prompt > args.max_model_len or tok_negative_prompt > args.max_model_len:
                print(f"Skipping prompt {idx} with positive length: {tok_positive_prompt} and negative length: {tok_negative_prompt}")
            
            else:
                translation_prompts.extend([positive_prompt, negative_prompt])
                row_ids.extend(["ultrafeedback_" + str(idx), "ultrafeedback_" + str(idx)])
                turn_ids.extend(["chosen", "rejected"])
                inputs_list.extend([prompt, prompt])
                chosen_list.extend([chosen_string, chosen_string])
                rejected_list.extend([rejected_string, rejected_string])
                source_list.extend([source, source])
        
        print("Length of translation prompts: ", len(translation_prompts))

        hf_dataset = datasets.Dataset.from_dict({
            "en_input": inputs_list, 
            "id": row_ids, 
            "en_chosen": chosen_list,
            "en_rejected": rejected_list,
            "source": source_list,
            "selection": turn_ids, 
            "translation_prompt": translation_prompts
        })

        # select the first 10 rows for testing
        hf_dataset = hf_dataset.select(range(args.filter_start, args.filter_end))
        print(f"Selected {len(hf_dataset)} rows for inference from {args.filter_start} to {args.filter_end}...")

        # Convert the Huggingface dataset to Ray Data.
        ds = ray.data.from_huggingface(hf_dataset)

        # Apply batch inference for all input data.
        ds = ds.repartition(10, shuffle=False)

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
            prompt = output["translation_prompt"]
            generated_text = output["generated_text"]
            output_dict[idx] = {"translation_prompt": prompt, "generated_text": generated_text}
            output_dict[idx].update(
                {"id": output["id"], 
                 "en_chosen": output["en_chosen"], 
                 "en_rejected": output["en_rejected"], 
                 "en_input": output["en_input"],
                 "selection": output["selection"],
                 "source": output["source"]})

        # Save the results in JSONL format.
        save_results(args.output_dir, output_dict, f"{args.filename}-final-{args.filter_start}-{args.filter_end}.jsonl")