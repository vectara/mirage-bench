from vllm import LLM, SamplingParams
import ray
from transformers import AutoTokenizer
import argparse, datasets
from datasets import Dataset
from typing import Dict
from tqdm import tqdm
import json, re, os, csv
import numpy as np
import pandas as pd

ISO_TO_LANG = {
    "ar": "arabic",
    "bn": "bengali",
    "hi": "hindi",
    "en": "english",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "ru": "russian",
    "es": "spanish",
    "zh": "chinese",
    "th": "thai",
    "ja": "japanese",
    "sw": "swahili",
    "yo": "yoruba",
    "fa": "persian",
    "id": "indonesian",
    "ko": "korean",
    "te": "telugu",
}

COHERENCE_PROMPT = """
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
{{Question}}\n
Documents in {{language}}:
{{Documents}}\n
Summary:
{{Summary}}\n
Rate the coherence of the summary on a scale of 1 to 5 and provide an explanation for your rating. Please use the format of: ##Rating: {rating} ##Explanation: {explanation}.
"""

# Load the predictions and references
def parse_answer(rag_answer: str) -> Dict[str, str]:
    
    context_string, answer_string = "", ""
    context = re.search(r'Reason(.*?)Answer:', rag_answer, re.DOTALL)
    answer = re.search(r'Answer:(.*?)$', rag_answer, re.DOTALL)
    
    if context:
        # get the citations from the context
        context_string = context.group(1).strip().replace("##", "").replace(":", "").strip()
    
    if answer:
        answer_string = answer.group(1).strip().split("\n\n")[0].replace("##", "").replace(":", "").strip()
        
    return "Context: " + context_string + " Answer: " + answer_string

def postprocess_coherence(answer: str) -> str:
    rating = 0
    coherence = re.search(r'Rating(.*?)Explanation:', answer, re.DOTALL)
    if coherence:
        rating = coherence.group(1).strip()
        rating = rating.replace(":", "").replace("#", '').strip()
        try:
            return (int(rating), False)
        except:
            print(f"Rating is not an integer: {rating} Answer: {answer}")
            return (0, True)
    else:
        print(f"Rating not found in answer: {answer}")
        return (0, True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_hf_dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--language_code", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    parser.add_argument("--output_filepath", type=str, default="context_grounding_nli_model_scores.tsv", required=False)
    parser.add_argument("--raw_output_scores", type=str, default="", required=False)
    parser.add_argument("--llm_judge", default="meta-llama/Meta-Llama-3-8B-Instruct", required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--max_new_tokens", type=int, default=2048, required=False)
    parser.add_argument("--temperature", type=float, default=0.1, required=False)
    parser.add_argument("--max_model_len", required=False, type=int, default=4096)
    parser.add_argument("--add_new_line", action="store_true", required=False)
    parser.add_argument("--top_k_sampling", type=int, default=5, required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading Lang: {args.language_code}")
    # Load the predictions and references

    load_dataset = datasets.load_dataset(args.eval_hf_dataset, args.language_code, split=args.split, cache_dir=args.cache_dir)
    hf_dataset = {}
    
    for row in load_dataset:
        hf_dataset[row["query_id"]] = row
    
    # Create a sampling params object.
    print(f"\tLoading LLM Judge: {args.llm_judge}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_judge)    
    terminators, stop_strings = [], []
    if "llama-3" in args.llm_judge.lower():
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        stop_strings = ["<|eot_id|>"]

    sampling_params = SamplingParams(temperature=args.temperature,
                                    max_tokens=args.max_new_tokens,
                                    stop_token_ids=terminators if terminators else None,
                                    stop=stop_strings if stop_strings else None, 
                                    )

    # Create a class to do batch inference.
    class LLMPredictor:

        def __init__(self):
            # Create an LLM.
                self.llm = LLM(model=args.llm_judge, 
                            max_model_len=args.max_model_len, 
                            max_num_seqs=1, 
                            max_seq_len_to_capture=args.max_model_len,
                            download_dir=args.cache_dir,
                            dtype="bfloat16", 
                            trust_remote_code=True)  # skip graph capturing for faster cold starts)

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            # Generate texts from the prompts.
            # The output is a list of RequestOutput objects that contain the prompt,
            # generated text, and other information.
            outputs = self.llm.generate(batch["prompt"], sampling_params)
            prompt = []
            generated_text = []
            for output in outputs:
                prompt.append(output.prompt)
                generated_text.append(' '.join([o.text for o in output.outputs]))
            
            return {
                "output": generated_text,
                "query_id": batch["query_id"], 
                "model_name": batch["model_name"],
            }
    

    # Start the evaluation
    model_names = []
    for output in list(hf_dataset.values())[0]["outputs"]:
        model_names.append(output["model"])
    
    results_judge = {query_id: {} for query_id in hf_dataset}
    raw_response_judge = {query_id: {} for query_id in hf_dataset}

    avg_mean_results_judge = {model_name: 0.0 for model_name in model_names}

    query_ids, model_names_final, prompts = [], [], []

    for query_id in tqdm(hf_dataset, desc="Processing queries", total=len(hf_dataset)):        
        for model_name in model_names:
            results_judge[query_id][model_name] = 0.0
            raw_response_judge[query_id][model_name] = ""
        
        prompt = hf_dataset[query_id]["prompt"]
        query = re.search(r'Question:(.*?)Contexts:', prompt, re.DOTALL)
        documents = re.search(r'Contexts:(.*?)Instruction:', prompt, re.DOTALL)
        
        if query: query = query.group(1).strip()
        if documents: documents = documents.group(1).strip()
        
        # response to evaluate
        for model_name in model_names:
            for model_output in hf_dataset[query_id]["outputs"]:
                if model_output["model"] == model_name:
                    summary = parse_answer(model_output["output"])

                    prompt = COHERENCE_PROMPT.replace("{{language}}", ISO_TO_LANG[args.language_code].capitalize())
                    prompt = prompt.replace("{{Question}}", query)
                    prompt = prompt.replace("{{Documents}}", documents)
                    prompt = prompt.replace("{{Summary}}", summary)

                    messages = [{"role": "user", "content": f"{prompt}"}]
                    prompt_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt_template)
                    query_ids.append(query_id)
                    model_names_final.append(model_name)
    
    # convert the list of prompts, query_ids and model_names into a huggingface dataset
    print(len(prompts), len(query_ids), len(model_names_final))

    dataset = {
        "prompt": prompts,
        "query_id": query_ids,
        "model_name": model_names_final,
    }

    hf_dataset_prompt = Dataset.from_dict(dataset)

    # Convert the Huggingface dataset to Ray Data.
    ds = ray.data.from_huggingface(hf_dataset_prompt)

    # Apply batch inference for all input data.
    ds = ds.repartition(12, shuffle=False)

    ds = ds.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=args.concurrency,
        # Specify the number of GPUs required per LLM instance.
        # NOTE: Do NOT set `num_gpus` when using vLLM with tensor-parallelism
        # (i.e., `tensor_parallel_size`).
        num_gpus=args.num_gpus,
        # Specify the batch size for inference.
        batch_size=args.batch_size,
        zero_copy_batch=True,
    )

    # Peek first 10 results.
    # NOTE: This is for local testing and debugging. For production use case,
    # one should write full result out as shown below.
    outputs = ds.take_all()

    output_dict = {}

    for idx, output in enumerate(outputs):
        generated_text = output["output"]
        (rating, error) = postprocess_coherence(generated_text)
        query_id, model_name = output["query_id"], output["model_name"]
        results_judge[query_id][model_name] = rating
        raw_response_judge[query_id][model_name] = generated_text
        avg_mean_results_judge[model_name] += rating
    
    # average of the llm judge ratings
    for model_name in model_names:
        avg_mean_results_judge[model_name] /= len(hf_dataset)

    # save the results in the original huggingface dataset
    hf_documents = []
    for query_id in hf_dataset:
        data = hf_dataset[query_id]
        llm_name = args.llm_judge.split("/")[-1]
        data[f"{llm_name}-fluency"] = results_judge[query_id]
        hf_documents.append(data)
    
    # save the results in the original huggingface dataset
    hf_dataset_new = datasets.Dataset.from_pandas(pd.DataFrame(hf_documents))
    print(f"Total documents in {args.language_code}: {len(hf_documents)}")
    hf_dataset_new.push_to_hub(args.eval_hf_dataset, config_name=args.language_code, private=False, split=args.split)  

    # store results in a csv file
    if not os.path.isfile(args.output_filepath):
        os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
        with open(args.output_filepath, "w", newline="\n") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["Model", "Language", "Reference", "Average"])

    os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
    with open(args.output_filepath, "a", newline="\n") as f:
        writer = csv.writer(f, delimiter="\t")
        for model_name in model_names:
            writer.writerow([model_name, 
                            args.language_code, 
                            args.llm_judge,
                            round(avg_mean_results_judge[model_name], 3)])
        f.write("\n\n")
        
    
    # save all ratings + llm_output separately in a jsonl file
    os.makedirs(os.path.dirname(args.raw_output_scores), exist_ok=True)
    with open(args.raw_output_scores, "w", encoding="utf-8") as f:
        for query_id in raw_response_judge:
            data = {
                "query_id": query_id,
                "ratings": results_judge.get(query_id, []),
                "outputs": raw_response_judge.get(query_id, [])
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
