"""
export CUDA_VISIBLE_DEVICES=0 # or any other GPU configuration
export NCCL_P2P_DISABLE=1
export HF_HOME=<your_cache_dir>
export DATASETS_HF_HOME=<your_cache_dir>

for lang in en; do
    python vllm_generation.py --language $lang --split dev \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset_name "nthakur/mirage-bench" \
    --cache_dir "<your_cache_dir>" \
    --temperature 0.1 \
    --max_new_tokens 2048 \
    --max_model_len 4096 \
    --batch_size 16 \
    --tensor_parallel_size 1 \
    --num_instances 1 \
    --max_num_seqs 1
done

NOTE: To use multiple GPUs for inference, use export CUDA_VISIBLE_DEVICES=0,1,2
and set --tensor_parallel_size to the number of GPUs e.g. 3
"""

import argparse
import logging

from mirage_bench import LoggingHandler, util
from mirage_bench.generate import VLLMClient

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    parser.add_argument("--model_name_or_path", default="meta-llama/Meta-Llama-3-8B-Instruct", required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--max_new_tokens", type=int, default=2048, required=False)
    parser.add_argument("--temperature", type=float, default=0.1, required=False)
    parser.add_argument("--dtype", type=str, default="bfloat16", required=False)
    parser.add_argument("--max_model_len", required=False, type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--num_instances", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--prompt_key", type=str, default="prompt", required=False)
    args = parser.parse_args()

    logging.info(f"Loading the VLLM client with {args.model_name_or_path}")
    vllm_client = VLLMClient(
        model_name_or_path=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        cache_dir=args.cache_dir,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        prompt_key=args.prompt_key,
        trust_remote_code=True,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_num_seqs=args.max_num_seqs,
    )

    # Load the documents & the dataset
    prompts_dict = util.load_prompts(dataset_name=args.dataset_name, language_code=args.language, split=args.split)
    logging.info(f"Loading {len(prompts_dict)} prompts for generation...")
    query_ids, prompts = list(prompts_dict.keys()), list(prompts_dict.values())

    # Generate the outputs for the prompts using the VLLM client
    outputs = vllm_client.batch_call(
        prompts=prompts,
        query_ids=query_ids,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        shards=args.shards,
    )

    # Save the output predictions}
    predictions = {}
    for output in outputs:
        query_id = output["query_id"]
        prediction = output["output"]
        predictions[query_id] = prediction

    ### print the top 5 predictions
    for query_id in query_ids[:5]:
        logging.info(f"Query ID: {query_id}")
        logging.info(f"Prompt: {prompts_dict[query_id]}")
        logging.info(f"{args.model_name_or_path} Prediction: {predictions[query_id]}\n")
