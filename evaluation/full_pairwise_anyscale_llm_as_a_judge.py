import requests
import argparse, datasets
from typing import Dict, Tuple
from transformers import AutoTokenizer
from tqdm import tqdm
import re, os, csv
import numpy as np
import pandas as pd
import random
import json, time

random.seed(42)

LLM_AS_A_JUDGE_PAIRWISE_PROMPT = """
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
{query}

[Reference Documents]
{documents}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

class AnyScaleAPI:
    def __init__(self, api_key: str, api_base: str, model: str):
        self.api_key = api_key
        self.api_base = api_base
        self.url = f"{self.api_base}/chat/completions"
        self.s = requests.Session()
        self.model = model
    
    def __call__(self, prompt: str, temperature: float = 0.3, max_tokens: int = 50):
        try:
            body = {
                "model": self.model,
                "messages": [{"role": "user", "content": f"{prompt}"}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            with self.s.post(self.url, headers={"Authorization": f"Bearer {self.api_key}"}, json=body) as resp:
                return resp.json()["choices"][0]["message"]["content"]
        
        except Exception as e:
            print(f"Error: {e}")
            return self.__call__(prompt, temperature, max_tokens)


def load_queries(hf_dataset: str) -> Dict[str, str]:
    queries = {}
    
    for query_id in hf_dataset:
        data = hf_dataset[query_id]
        query = data["prompt"].split("Question:")[1].split("\n\n")[0].strip()
        queries[query_id] = query
    return queries

def load_documents(hf_dataset: str, model_name: str) -> Dict[str, str]:
    documents_dict = {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    for query_id in hf_dataset:
        documents_dict[query_id] = {}
        prompt = hf_dataset[query_id]["prompt"]
        context = prompt.split("\n\nContexts:")[1].split("\n\nInstruction")[0].strip()
        documents = context.split("\n")
        for document in documents:
            doc_id = document.split("]")[0].replace("[", "").strip()
            doc_text = "]".join(document.split("]")[1:]).strip()
            tokens = tokenizer.tokenize(doc_text)[:512]
            doc_text = tokenizer.convert_tokens_to_string(tokens)
            documents_dict[query_id][doc_id] = doc_text
    
    return documents_dict

# Load the predictions and references
def parse_rag_answer(rag_answer: str) -> Dict[str, str]:
    
    context_string, answer_string = "", ""
    context = re.search(r'Reason(.*?)Answer:', rag_answer, re.DOTALL)
    answer = re.search(r'Answer:(.*?)$', rag_answer, re.DOTALL)
    
    if context:
        # get the citations from the context
        context_string = context.group(1).strip().replace("##", "").replace(":", "").strip()
    
    if answer:
        answer_string = answer.group(1).strip().split("\n\n")[0].replace("##", "").replace(":", "").strip()
        
    return "Context: " + context_string + " Answer: " + answer_string

def postprocess_pairwise(output: str) -> Tuple[str, bool]:
    try:
        rating = re.search(r"\[\[(.*?)\]\]", output).group(1)
        return rating, False
    except:
        return "", True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_hf_dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--reference_model", type=str, required=True)
    parser.add_argument("--language_code", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    parser.add_argument("--output_filepath", type=str, default="avg_openai_llm_as_a_judge_scores.tsv", required=False)
    parser.add_argument("--raw_output_scores", type=str, default="", required=False)
    parser.add_argument("--raw_input_scores", type=str, default="", required=False)
    parser.add_argument("--temperature", type=float, default=0.1, required=True)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--max_tokens", type=int, default=1024, required=False)
    parser.add_argument("--sample_queries", type=int, default=100, required=False)

    args = parser.parse_args()

    print(f"Loading Lang: {args.language_code}")
    # Load the predictions and references

    load_dataset = datasets.load_dataset(args.eval_hf_dataset, args.language_code, split=args.split, cache_dir=args.cache_dir)
    hf_dataset = {}

    for row in load_dataset:
        hf_dataset[row["query_id"]] = row
    
    # load queries and relevant documents
    hf_queries = load_queries(hf_dataset)
    hf_documents = load_documents(hf_dataset, args.reference_model)
    
    print(f"\tLoading LLM Judge: {args.reference_model}")
    api_base = os.getenv("ANYSCALE_BASE_URL")
    token = os.getenv("ANYSCALE_API_KEY")
    anyscale_api_judge = AnyScaleAPI(token, api_base, args.reference_model)

    model_names = []
    for output in list(hf_dataset.values())[0]["outputs"]:
        model_names.append(output["model"])
    
    results_pairwise_winrate = {query_id: {} for query_id in hf_dataset}
    raw_response_pairwise_judge = {query_id: {} for query_id in hf_dataset}
    final_sorted_results = {query_id: {} for query_id in hf_dataset}
    
    if len(hf_queries) > args.sample_queries:
        query_ids_sampled = set()
        if os.path.exists(args.raw_input_scores):
            with open(args.raw_input_scores, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    query_id = data["query_id"]
                    query_ids_sampled.add(query_id)
        
        if len(query_ids_sampled) < args.sample_queries:
            query_sample = [query_id for query_id in hf_queries if query_id not in query_ids_sampled]

        queries_added = args.sample_queries - len(query_ids_sampled)
        print(f"Already sampled {len(query_ids_sampled)} queries. Adding {queries_added} queries")
        hf_queries_final = {query_id: hf_queries[query_id] for query_id in query_ids_sampled} 
        query_sample = random.sample(query_sample, queries_added)
        hf_queries_final.update({query_id: hf_queries[query_id] for query_id in query_sample})

    for query_id in tqdm(hf_queries_final, desc="Processing queries", total=len(hf_queries_final)):
        # Do pairwise-comparisons based on the absolute rating
        win_rates = {model_name: 0 for model_name in model_names}
        random.shuffle(model_names)

        rag_answers = {}
        for model_output in hf_dataset[query_id]["outputs"]:
            rag_answers[model_output["model"]] = parse_rag_answer(model_output["output"])

        
        all_pairwise_prompts, all_model_combinations = [], []
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model_i, model_j = model_names[i], model_names[j]
                query_text = hf_queries_final[query_id]
                documents = "\n".join([f"[{doc_id}]: {doc_text}" for doc_id, doc_text in hf_documents[query_id].items()])
                if random.choice([True, False]): model_i, model_j = model_j, model_i
                answer_i, answer_j = rag_answers[model_i], rag_answers[model_j]
                
                prompt = LLM_AS_A_JUDGE_PAIRWISE_PROMPT.replace("{query}", query_text)
                prompt = prompt.replace("{documents}", documents)
                prompt = prompt.replace("{answer_a}", answer_i)
                prompt = prompt.replace("{answer_b}", answer_j)
                all_pairwise_prompts.append(prompt)
                all_model_combinations.append((model_i, model_j))

        # batch call all pairwise prompts
        responses = []
        for prompt in tqdm(all_pairwise_prompts, desc="Calling Anyscale API", total=len(all_pairwise_prompts), leave=False):
            response = anyscale_api_judge.__call__(prompt, temperature=args.temperature, max_tokens=args.max_tokens)
            responses.append(response)
        
        for response, (model_i, model_j) in zip(responses, all_model_combinations):
            rating, error = postprocess_pairwise(response)
            raw_response_pairwise_judge[query_id][model_i + "," + model_j] = response
            
            if rating == "A":
                win_rates[model_i] += 1
            elif rating == "B":
                win_rates[model_j] += 1
            elif rating == "C":
                win_rates[model_i] += 0.5
                win_rates[model_j] += 0.5

        results_pairwise_winrate[query_id] = win_rates
        
        # compute the final order
        final_ranking_order = []
        
        winrates_sorted = sorted(win_rates, key=lambda x: win_rates[x], reverse=True)
        final_ranking_order.extend(winrates_sorted)
        final_sorted_results[query_id] = final_ranking_order
        
        os.makedirs(os.path.dirname(args.raw_output_scores), exist_ok=True)
        with open(args.raw_output_scores, "a", encoding="utf-8") as f:
            data = {
                "query_id": query_id,
                "pairwise_outputs": raw_response_pairwise_judge[query_id],
                "pairwise_winrates": results_pairwise_winrate.get(query_id, ""),
                "final_order": final_ranking_order
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
