"""
export HF_HOME=/mnt/users/n3thakur/cache
export DATASETS_HF_HOME=/mnt/users/n3thakur/cache

for lang in en; do
    python evaluate_answer_rouge_blue.py --language $lang --split dev \
    --dataset_name "nthakur/mirage-eval" \
    --prediction_dataset "nthakur/mirage-eval-rag-output" \
    --prediction_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --reference_model "gpt-4-azure"
done
"""

import argparse
import logging

from mirage_bench import LoggingHandler, util
from mirage_bench.evaluate import RougeBleuEvaluator

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
    parser.add_argument("--reference_model", default="gpt-4-azure", help="Reference model to compute the scores")
    args = parser.parse_args()

    # Load the evaluator
    evaluator = RougeBleuEvaluator(language_code=args.language)

    # Load the documents & the dataset
    documents = util.load_documents(dataset_name=args.dataset_name, language_code=args.language, split=args.split)

    # Load predictions available
    predictions = util.load_predictions(
        dataset_name=args.prediction_dataset,
        language_code=args.language,
        split=args.split,
        model_name=args.prediction_model,
    )

    # Load the reference predictions
    reference_predictions = util.load_predictions(
        dataset_name=args.prediction_dataset,
        language_code=args.language,
        split=args.split,
        model_name=args.reference_model,
    )

    # Evaluate the predictions
    scores = evaluator.evaluate(
        predictions=predictions, reference_predictions=reference_predictions, documents=documents
    )
