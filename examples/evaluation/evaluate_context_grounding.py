"""
export CUDA_VISIBLE_DEVICES=5
export HF_HOME=/mnt/users/n3thakur/cache
export DATASETS_HF_HOME=/mnt/users/n3thakur/cache

for lang in en; do
    python evaluate_context_grounding.py --language $lang --split dev \
    --judge "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" \
    --dataset_name "nthakur/mirage-eval" \
    --prediction_dataset "nthakur/mirage-eval-rag-output" \
    --prediction_model "meta-llama/Meta-Llama-3.1-70B-Instruct"
done
"""

import argparse
import logging

from mirage_bench import LoggingHandler, util
from mirage_bench.evaluate import ContextGroundingEvaluator

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
    parser.add_argument("--judge", default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    parser.add_argument("--prediction_dataset", default=None)
    parser.add_argument("--prediction_model", default=None)
    args = parser.parse_args()

    # Load the evaluator
    evaluator = ContextGroundingEvaluator(language_code=args.language, model_name=args.judge)

    # Load the documents & the dataset
    documents = util.load_documents(dataset_name=args.dataset_name, language_code=args.language, split=args.split)

    # Load predictions available
    predictions = util.load_predictions(
        dataset_name=args.prediction_dataset,
        language_code=args.language,
        split=args.split,
        model_name=args.prediction_model,
    )

    # Evaluate the predictions
    scores = evaluator.evaluate(output=predictions, documents=documents)
