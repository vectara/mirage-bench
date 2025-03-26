"""
export HF_HOME=<your_cache_dir>
export DATASETS_HF_HOME=<your_cache_dir>
export AZURE_OPENAI_API_VERSION=XXXX
export AZURE_OPENAI_ENDPOINT=XXXX
export AZURE_OPENAI_API_KEY=XXXX

PREDICTION_MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Meta-Llama-3-70B-Instruct"
)

for lang in en; do
    python pairwise_eval_openai.py --language $lang --split dev \
    --client "azure_openai" \
    --judge "gpt-4o-mini" \
    --dataset_name "nthakur/mirage-bench" \
    --prediction_dataset "nthakur/mirage-bench-output" \
    --all_models "${PREDICTION_MODELS[@]}" \
    --cache_dir "<your_cache_dir>" \
    --temperature 0.1 \
    --max_new_tokens 2048 \
    --max_model_len 4096 \
    --batch_size 1 \
    --sample_queries 5
done
"""

import argparse
import logging

from mirage_bench import LoggingHandler, util
from mirage_bench.evaluate import PairwiseLLMJudgeEvaluator

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

PAIRWISE_LLM_JUDGE_PROMPT = """
Please act as an impartial judge and evaluate the quality of the responses provided \
by two AI assistants tasked to answer the question displayed below, based on a set \
of documents retrieved by a search engine.
You should choose the assistant that best answers the user question based on a set \
of reference documents that may or not be relevant referenced in the IEEE format.\n \
Your evaluation should consider factors such as the correctness, helpfulness, completeness, accuracy, depth, and level of detail of their responses.
Details are only useful if they answer the user question. If an answer \
contains non-relevant details, it should not be preferred over one that only \
use relevant information.
Begin your evaluation by explaining why each answer correctly answers the user \
question. Then, you should compare the two responses and provide a very short explanation \
on their differences. Avoid any position biases and ensure that the order in which \
the responses were presented does not influence your decision. Do not allow the \
length of the responses to influence your evaluation. Be as objective as possible.
After providing your explanation, output your final verdict by strictly following \
this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, \
and "[[C]]" for a tie.

[User Question]
{{query}}

[Reference Documents]
{{documents}}

[The Start of Assistant A's Answer]
{{answer_a}}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{{answer_b}}
[The End of Assistant B's Answer]
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--prediction_dataset", default=None)
    parser.add_argument("--all_models", nargs="+", default=None)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    parser.add_argument("--client", type=str, default="openai", required=False)
    parser.add_argument("--judge", default="gpt-4o-mini", required=False)
    parser.add_argument("--batch_size", type=int, default=1, required=False)
    parser.add_argument("--max_new_tokens", type=int, default=2048, required=False)
    parser.add_argument("--temperature", type=float, default=0.1, required=False)
    parser.add_argument("--dtype", type=str, default="bfloat16", required=False)
    parser.add_argument("--max_model_len", required=False, type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--num_instances", type=int, default=1)
    parser.add_argument("--sample_queries", type=int, default=1, required=False)

    args = parser.parse_args()

    # Load the evaluator
    evaluator = PairwiseLLMJudgeEvaluator(
        client=args.client,
        model_name_or_path=args.judge,
    )

    # Load the documents & the dataset
    documents = util.load_documents(dataset_name=args.dataset_name, language_code=args.language, split=args.split)
    queries = util.load_queries(dataset_name=args.dataset_name, language_code=args.language, split=args.split)

    # Load predictions available
    if args.all_models:
        predictions = {}
        ### predictions = {model_name: {query_id_1: rag_output_1, query_id_2: rag_output_2, ...}, model_name_2: {...}}
        for model_name in args.all_models:
            predictions[model_name] = util.load_predictions(
                dataset_name=args.prediction_dataset,
                language_code=args.language,
                split=args.split,
                model_name=model_name,
            )

    # Evaluate the predictions
    scores = evaluator.evaluate(
        predictions=predictions,
        all_model_names=args.all_models,
        documents=documents,
        queries=queries,
        prompt=PAIRWISE_LLM_JUDGE_PROMPT,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        sample_max=args.sample_queries,
    )
