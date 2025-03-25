"""
export HF_HOME=<your_cache_dir>
export DATASETS_HF_HOME=<your_cache_dir>

for lang in en; do
    python evaluate_context_map_recall.py --language $lang --split dev \
    --dataset_name "nthakur/mirage-bench" \
    --prediction_dataset "nthakur/mirage-bench-output" \
    --prediction_model "meta-llama/Meta-Llama-3-8B-Instruct"
done
"""

import argparse
import logging

from mirage_bench import LoggingHandler, util
from mirage_bench.evaluate import ContextMAPRecallEvaluator

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
    args = parser.parse_args()

    # Load the evaluator
    evaluator = ContextMAPRecallEvaluator(language_code=args.language, k_values=[10])

    # Load the documents & the dataset
    documents = util.load_documents(dataset_name=args.dataset_name, language_code=args.language, split=args.split)
    qrels = util.load_qrels(dataset_name=args.prediction_dataset, language_code=args.language, split=args.split)

    # Load predictions available
    predictions = util.load_predictions(
        dataset_name=args.prediction_dataset,
        language_code=args.language,
        split=args.split,
        model_name=args.prediction_model,
    )

    # Evaluate the predictions
    scores = evaluator.evaluate(predictions=predictions, documents=documents, qrels=qrels)
