"""
export CUDA_VISIBLE_DEVICES=-1
export HF_HOME=/mnt/users/n3thakur/cache
export DATASETS_HF_HOME=/mnt/users/n3thakur/cache

for lang in en; do
    python evaluate_reranker_score.py --language $lang --split dev \
    --dataset_name "nthakur/mirage-eval" \
    --prediction_dataset "nthakur/mirage-eval-rag-output" \
    --prediction_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --judge "BAAI/bge-reranker-v2-m3" \
    --cache_dir "/mnt/users/n3thakur/cache"
done
"""

import argparse
import logging

from mirage_bench import LoggingHandler, util
from mirage_bench.evaluate import RerankerScoreEvaluator

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
    parser.add_argument("--prediction_model", default=None)
    parser.add_argument("--judge", default="BAAI/bge-reranker-v2-m3", help="Reranker model to compute the scores")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--use_fp16", action="store", default=False, help="Use FP16 for evaluation")
    args = parser.parse_args()

    # Load the evaluator
    evaluator = RerankerScoreEvaluator(model_name=args.judge, cache_dir=args.cache_dir, use_fp16=args.use_fp16)

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
    scores = evaluator.evaluate(predictions=predictions, documents=documents, queries=queries)
