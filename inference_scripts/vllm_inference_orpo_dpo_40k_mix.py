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
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Union, Optional


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
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        temp_model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)
        del temp_model

        # Create a sampling params object.
        terminators, stop_strings = [], []
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
                            trust_remote_code=True, # skip graph capturing for faster cold starts
                            )

            def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
                # Generate texts from the prompts.
                # The output is a list of RequestOutput objects that contain the prompt,
                # generated text, and other information.
                outputs = self.llm.generate(batch["prompt_llama"], sampling_params)
                prompt = []
                generated_text = []
                for output in outputs:
                    prompt.append(output.prompt)
                    generated_text.append(' '.join([o.text for o in output.outputs]))
                
                return {
                    "translation_prompt": prompt,
                    "generated_text": generated_text,
                    "id": batch["id"],
                    "turn": batch["turn"],
                    "en_chosen": batch["en_chosen"],
                    "en_prompt": batch["en_prompt"],
                    "en_rejected": batch["en_rejected"],
                    "selection": batch["selection"],
                    "source": batch["source"],
                }


        # Read one text file from S3. Ray Data supports reading multiple files
        # from cloud storage (such as JSONL, Parquet, CSV, binary format).
        # ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
        hf_dataset = datasets.load_dataset(args.dataset_name, args.language, split=args.split, cache_dir=args.cache_dir)

        # select the first 10 rows for testing
        hf_dataset = hf_dataset.select(range(args.filter_start, args.filter_end))

        prompts = []
        for row in hf_dataset:
            prompt = row["translation_prompt"]
            system = prompt.split("Please translate the given question")[0].strip()
            user_prompt = prompt.replace(system, "").strip()
            user_prompt = user_prompt.split("Please keep in mind that:")[0] + "\n\n" + "Please keep in mind that:" + user_prompt.split("Please keep in mind that:")[1]
            messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},]

            prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )
            prompts.append(prompt)
        
        hf_dataset = hf_dataset.add_column("prompt_llama", prompts)
        print(f"Selected {len(hf_dataset)} rows for inference from {args.filter_start} to {args.filter_end}...")

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
            prompt = output["translation_prompt"]
            generated_text = output["generated_text"]
            output_dict[idx] = {"translation_prompt": prompt, "generated_text": generated_text}
            output_dict[idx].update(
                {"id": output["id"], 
                 "turn": output["turn"],
                 "en_chosen": output["en_chosen"], 
                 "en_rejected": output["en_rejected"], 
                 "en_prompt": output["en_prompt"],
                 "selection": output["selection"],
                 "source": output["source"]})

        # Save the results in JSONL format.
        save_results(args.output_dir, output_dict, f"{args.filename}-final-{args.filter_start}-{args.filter_end}.jsonl")