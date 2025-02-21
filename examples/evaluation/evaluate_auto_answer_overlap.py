"""
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_P2P_DISABLE=1
export HF_HOME=/mnt/users/n3thakur/cache
export DATASETS_HF_HOME=/mnt/users/n3thakur/cache

for lang in en; do
    python evaluate_auto_answer_overlap.py --language $lang --split dev \
    --judge "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset_name "nthakur/mirage-eval" \
    --prediction_dataset "nthakur/mirage-eval-rag-output" \
    --prediction_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --reference_model "gpt-4-azure" \
    --cache_dir "/mnt/users/n3thakur/cache" \
    --temperature 0.1 \
    --max_new_tokens 2048 \
    --max_model_len 4096 \
    --batch_size 16 \
    --tensor_parallel_size 4 \
    --num_instances 1
done
"""

import argparse
import logging

from mirage_bench import LoggingHandler, util
from mirage_bench.evaluate import AutomaticAnswerOverlapEvaluator

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

ANSWER_OVERLAP_PROMPT = """
You are an AI assistant. In the following task, you are given a Question, a RAG application's response, and a Ground-truth Answer referred to as 'Label' in {{language}}.
Assess how well the RAG application's response aligns with the Label, using the grading rubric below:\n\n
1: The response is not aligned with the Label or is off-topic; includes hallucination.
2: The response admits it cannot provide an answer or lacks context; honest.
3: The response is relevant but contains notable discrepancies or inaccuracies.
4: The response is acceptable, sufficient but not exhaustive.
5: The response is fully accurate and comprehensive, based on the Label.

Treat the Label as the definitive answer. Present your justification followed by your final score in the format: "[[score]]",
----
Example:
Justification: The response partially aligns with the label but with some discrepancies. Score: [[3]]
-----
Question in {{language}}:
{{question}}\n

Label in {{language}}:
{{label}}\n

RAG Application Response in {{language}}:
{{response}}\n
Treat the label as the definitive answer. Present your justification in English and followed by your final score in the format: "[[score]]",
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--prediction_dataset", default=None)
    parser.add_argument("--prediction_model", default=None)
    parser.add_argument("--reference_model", default="gpt-4-azure", help="Reference model to compute the scores")
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    parser.add_argument("--judge", default="meta-llama/Meta-Llama-3-8B-Instruct", required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--max_new_tokens", type=int, default=2048, required=False)
    parser.add_argument("--temperature", type=float, default=0.1, required=False)
    parser.add_argument("--dtype", type=str, default="bfloat16", required=False)
    parser.add_argument("--max_model_len", required=False, type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--num_instances", type=int, default=1)
    args = parser.parse_args()

    # Load the evaluator
    evaluator = AutomaticAnswerOverlapEvaluator(
        language_code=args.language,
        model_name_or_path=args.judge,
        tensor_parallel_size=args.tensor_parallel_size,
        cache_dir=args.cache_dir,
        max_length=args.max_model_len,
        max_num_seqs=1,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        answer_regex=r"Answer:(.*?)$",
        metric_name="answer_overlap",
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

    # Load predictions available
    reference_predictions = util.load_predictions(
        dataset_name=args.prediction_dataset,
        language_code=args.language,
        split=args.split,
        model_name=args.reference_model,
    )

    # Evaluate the predictions
    scores = evaluator.evaluate(
        predictions=predictions,
        reference_predictions=reference_predictions,
        documents=documents,
        queries=queries,
        prompt=ANSWER_OVERLAP_PROMPT,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        postprocess_regex=r"Score:(.*?)$",
    )
