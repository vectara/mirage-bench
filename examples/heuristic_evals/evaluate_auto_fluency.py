"""
export CUDA_VISIBLE_DEVICES=0,1,2,3 (or any other GPU configuration)
export NCCL_P2P_DISABLE=1
export HF_HOME=<your_cache_dir>
export DATASETS_HF_HOME=<your_cache_dir>

for lang in en; do
    python evaluate_auto_fluency.py --language $lang --split dev \
    --judge "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset_name "nthakur/mirage-bench" \
    --prediction_dataset "nthakur/mirage-bench-output" \
    --prediction_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --cache_dir "<your_cache_dir>" \
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
from mirage_bench.evaluate import AutomaticFluencyEvaluator

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

FLUENCY_PROMPT = """
You will be given one summary written for a question and documents from Wikipedia in {{language}}.
Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions carefully. Please keep this
document open while reviewing, and refer to it as needed.\n\n
Evaluation Criteria:
Coherence (1-5) - the collective quality of all sentences. We align this dimension with
the DUC quality question of structure and coherence whereby 'the summary should be
well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.'\n\n
Evaluation Steps:
1. Read the question and Wikipedia documents in {{language}} carefully and identify the main topic and key points.
2. Read the summary and check whether it answers the question. Check if the summary covers the main
topic and key points required to answer the question, and if it presents them in a clear and logical order.
3. Assign a rating for coherence on a scale of 1 to 5 and provide an explanation, where 1 is the lowest and 5 is the highest
based on the Evaluation Criteria.\n\n
Example:
Question in {{language}}:
{{question}}\n
Documents in {{language}}:
{{documents}}\n
Summary:
{{rag_answer}}\n
Provide an explanation and rate the coherence of the summary on a scale of 1 to 5 and provide an explanation for your rating.
Please use the format of: ##Explanation: <explanation> ##Rating: <rating>.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--prediction_dataset", default=None)
    parser.add_argument("--prediction_model", default=None)
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
    evaluator = AutomaticFluencyEvaluator(
        language_code=args.language,
        model_name_or_path=args.judge,
        tensor_parallel_size=args.tensor_parallel_size,
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
        prompt=FLUENCY_PROMPT,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        postprocess_regex=r"Rating:(.*?)$",
    )
