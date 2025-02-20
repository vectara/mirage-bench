"""
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/mnt/users/n3thakur/cache
export DATASETS_HF_HOME=/mnt/users/n3thakur/cache

for lang in en; do
    python evaluate_automatic_fluency.py --language $lang --split dev \
    --judge "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset_name "nthakur/mirage-eval" \
    --prediction_dataset "nthakur/mirage-eval-rag-output" \
    --cache_dir "/mnt/users/n3thakur/cache" \
    --temperature 0.1 \
    --max_new_tokens 2048 \
    --max_model_len 4096 \
    --batch_size 16 \
    --num_gpus 1 \
    --concurrency 4
done
"""

import argparse
import logging

from mirage_bench import LoggingHandler, util
from mirage_bench.evaluate import AutomaticFluencyEvaluator

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
    parser.add_argument("--prediction_dataset", default=None)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    parser.add_argument("--judge", default="meta-llama/Meta-Llama-3-8B-Instruct", required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--max_new_tokens", type=int, default=2048, required=False)
    parser.add_argument("--temperature", type=float, default=0.1, required=False)
    parser.add_argument("--dtype", type=str, default="bfloat16", required=False)
    parser.add_argument("--max_model_len", required=False, type=int, default=4096)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    # Load the evaluator
    evaluator = AutomaticFluencyEvaluator(
        language_code=args.language,
        model_name_or_path=args.judge,
        cache_dir=args.cache_dir,
        max_length=args.max_model_len,
        max_num_seqs=1,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Load the documents & the dataset
    documents = util.load_documents(dataset_name=args.dataset_name, language_code=args.language, split=args.split)
    queries = util.load_queries(dataset_name=args.dataset_name, language_code=args.language, split=args.split)

    # Load predictions available
    predictions = util.load_predictions(
        dataset_name=args.prediction_dataset,
        language_code=args.language,
        split=args.split,
        model_name=args.prediction_model,
    )

    # Evaluate the predictions
    scores = evaluator.evaluate(
        predictions=predictions,
        documents=documents,
        queries=queries,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        concurrency=args.concurrency,
    )
