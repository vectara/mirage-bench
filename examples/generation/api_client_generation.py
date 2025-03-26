"""
export HF_HOME=<your_cache_dir>
export DATASETS_HF_HOME=<your_cache_dir>
export AZURE_OPENAI_ENDPOINT="xxxxx"
export AZURE_OPENAI_API_KEY="xxxx"

for lang in en; do
    python api_client_generation.py --language $lang --split dev \
    --model gpt-4o-mini \
    --dataset_name "nthakur/mirage-bench" \
    --temperature 0.1 \
    --max_new_tokens 2048 \
    --max_model_len 4096
done
"""

import argparse
import logging

from mirage_bench import LoggingHandler, util
from mirage_bench.generate import (
    AzureOpenAIClient,
)

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
    parser.add_argument("--model_name_or_path", default="gpt-4o-mini", required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--max_new_tokens", type=int, default=2048, required=False)
    parser.add_argument("--temperature", type=float, default=0.1, required=False)
    parser.add_argument("--max_model_len", type=int, default=4096, required=False)
    parser.add_argument("--max_num_seqs", type=int, default=1, required=False)
    parser.add_argument("--prompt_key", type=str, default="prompt", required=False)
    args = parser.parse_args()

    logging.info(f"Loading the VLLM client with {args.model_name_or_path}")
    azure_openai_client = AzureOpenAIClient(model_name_or_path=args.model_name_or_path)

    #### Anyscale client ####
    # export ANYSCALE_BASE_URL="https://api.endpoints.anyscale.com/v1"
    # export ANYSCALE_API_KEY="xxxx"
    # anyscale_client = AnyScaleAPIClient(model_name_or_path=args.model_name_or_path)

    #### Claude OpenAI client ####
    # export ANTHROPIC_API_KEY="xxxx"
    # claude_client = ClaudeAPIClient(model_name_or_path=args.model_name_or_path)

    #### Cohere client ####
    # export COHERE_API_KEY="xxxx"
    # cohere_client = CohereAPIClient(model_name_or_path=args.model_name_or_path)

    #### Gemini client ####
    # export GOOGLE_API_KEY="xxxx"
    # gemini_client = GeminiAPIClient(model_name_or_path=args.model_name_or_path)

    #### OpenAI client ####
    # export OPENAI_API_KEY="xxxx"
    # export ORGANIZATION="xxxx"
    # export PROJECT_ID="xxxx"
    # openai_client = OpenAIClient(model_name_or_path=args.model_name_or_path)

    # Load the documents & the dataset
    prompts_dict = util.load_prompts(dataset_name=args.dataset_name, language_code=args.language, split=args.split)
    logging.info(f"Loading {len(prompts_dict)} prompts for generation...")
    query_ids, prompts = list(prompts_dict.keys()), list(prompts_dict.values())

    # Generate the outputs for the prompts using the VLLM client
    outputs = azure_openai_client.batch_call(
        prompts=prompts,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # Save the output predictions}
    predictions = {}
    for query_id, prediction in zip(query_ids, outputs):
        predictions[query_id] = prediction

    ### print the top 5 predictions
    for query_id in query_ids[:5]:
        logging.info(f"Query ID: {query_id}")
        logging.info(f"Prompt: {prompts_dict[query_id]}")
        logging.info(f"{args.model_name_or_path} Prediction: {predictions[query_id]}\n")
