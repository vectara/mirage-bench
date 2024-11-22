from openai import AzureOpenAI
import openai
import argparse, datasets
from typing import Dict, Tuple
from tqdm import tqdm
import re, os, csv
import numpy as np
import pandas as pd
import random
import json, time

random.seed(42)

LLM_AS_A_JUDGE_POINTWISE_PROMPT = """
### Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

### The instruction to evaluate:
{instruction}

### Response to evaluate:
{response}

### Reference Answer (Score 5):
{reference_answer}

### Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual. and/or cites relevant context.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual, and/or cites relevant context.
Score 3: The response is somewhat correct, accurate, and/or factual, and/or cites relevant context.
Score 4: The response is mostly correct, accurate, and factual, and cites relevant context.
Score 5: The response is completely correct, accurate, and factual and cites relevant context.

### Feedback:
"""

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
question. Then, you should compare the two responses and provide a short explanation \
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

class OpenAIClient:
    def __init__(self, 
                 model: str = None, 
                 endpoint: str = None, 
                 api_key: str = None, 
                 api_version: str = "2024-02-01", 
                 wait: int = 60):
        
        model = model.replace("-azure", "") if "-azure" in model else model
        self.deployment_name = "gpt-35-turbo" if "gpt-3.5-turbo" in model else model
        self.wait = wait
        self.client = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") if endpoint is None else endpoint, 
            api_key=os.getenv("AZURE_OPENAI_API_KEY") if api_key is None else api_key,  
            api_version=api_version,
        )
    
    def response(self, prompt: str, temperature: float, n: int):
        try:
            response = self.client.chat.completions.create(
                    model=self.deployment_name, # model = "deployment_name".
                    messages=[{"role": "user", "content": f"{prompt}"}],
                    temperature=temperature,
                    n=n,
            )
            output = response.choices[0].message.content
            return output
        
        except openai.RateLimitError as e:
            retry_time = int(str(e).split("retry after")[1].split('second')[0].strip())            
            time.sleep(retry_time)
            return self.response(prompt, temperature, n)
        
        except openai.InternalServerError as e:
            time.sleep(self.wait)
            return self.response(prompt, temperature, n)

    def __call__(self, prompts: str, temperature: float, n: int):
        responses = []

        for prompt in prompts:
            output = self.response(prompt, temperature, n)
            responses.append(output)
        
        return responses
    
    def batch_call(self, prompts, batch_size=1, name=None, **kwargs):
        batches = [
            prompts[i : i + batch_size]
            for i in range(0, len(prompts), batch_size)
        ]

        results = []
        for i, batch in enumerate(
            tqdm(batches, desc=f"Collecting {name} responses", leave=False)
        ):
            responses = self.__call__(batch, **kwargs)
            results.extend(responses)
        
        return results
        
def load_queries(hf_dataset: str) -> Dict[str, str]:
    queries = {}
    
    for query_id in hf_dataset:
        data = hf_dataset[query_id]
        query = data["prompt"].split("Question:")[1].split("\n\n")[0].strip()
        queries[query_id] = query
    return queries

def load_documents(hf_dataset: str) -> Dict[str, str]:
    documents_dict = {}

    for query_id in hf_dataset:
        documents_dict[query_id] = {}
        prompt = hf_dataset[query_id]["prompt"]
        context = prompt.split("\n\nContexts:")[1].split("\n\nInstruction")[0].strip()
        documents = context.split("\n")
        for document in documents:
            doc_id = document.split("]")[0].replace("[", "").strip()
            doc_text = "]".join(document.split("]")[1:]).strip()
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

def postprocess_rating(output: str) -> Tuple[int, bool]:
    try:
        rating = int(re.search(r"\[RESULT\] (\d)", output).group(1))
        return rating, False
    except:
        return 0, True

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
    parser.add_argument("--temperature", type=float, default=0.1, required=True)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--sample_queries", type=int, default=200, required=False)

    args = parser.parse_args()

    print(f"Loading Lang: {args.language_code}")
    # Load the predictions and references

    load_dataset = datasets.load_dataset(args.eval_hf_dataset, args.language_code, split=args.split, cache_dir=args.cache_dir)
    hf_dataset = {}

    for row in load_dataset:
        hf_dataset[row["query_id"]] = row
    
    # load queries and relevant documents
    hf_queries = load_queries(hf_dataset)
    hf_documents = load_documents(hf_dataset)
    
    print(f"\tLoading LLM Judge: {args.reference_model}")
    openai_judge = OpenAIClient(model=args.reference_model)

    model_names = []
    for output in list(hf_dataset.values())[0]["outputs"]:
        model_names.append(output["model"])
    
    results_judge = {query_id: {} for query_id in hf_dataset}
    raw_response_judge = {query_id: {} for query_id in hf_dataset}
    results_pairwise_winrate = {query_id: {} for query_id in hf_dataset}
    raw_response_pairwise_judge = {query_id: {} for query_id in hf_dataset}
    final_sorted_results = {query_id: {} for query_id in hf_dataset}

    if len(hf_queries) > args.sample_queries:
        hf_queries = dict(random.sample(hf_queries.items(), args.sample_queries))

    for query_id in tqdm(hf_queries, desc="Processing queries", total=len(hf_queries)):
        score_models = {5: set(), 4: set(), 3: set(), 2: set(), 1: set(), 0: set()}

        # store the answer for each model ooutput
        rag_answers = {}
        for model_output in hf_dataset[query_id]["outputs"]:
            rag_answers[model_output["model"]] = parse_rag_answer(model_output["output"])

        # response to evaluate
        all_pointwise_prompts, all_model_names = [], []
        for model_name in model_names:
            if model_name != args.reference_model:
                rag_answer = rag_answers[model_name]
                
                prompt = LLM_AS_A_JUDGE_POINTWISE_PROMPT.replace("{instruction}", hf_dataset[query_id]["prompt"])
                prompt = prompt.replace("{response}", rag_answer)
                prompt = prompt.replace("{reference_answer}", rag_answers[args.reference_model])
                all_pointwise_prompts.append(prompt)
                all_model_names.append(model_name)
        
        # batch call all pointwise prompts
        responses = openai_judge.batch_call(all_pointwise_prompts, 
                                            batch_size=args.batch_size, 
                                            temperature=args.temperature,
                                            name="pointwise",
                                            n=1)

        results_judge[query_id][args.reference_model] = 5
        raw_response_judge[query_id][args.reference_model] = ""
        score_models[5].add(args.reference_model)
        
        # batch evaluate each model and its response
        for model_name, response in zip(all_model_names, responses):  
            rating, error = postprocess_rating(response)
            results_judge[query_id][model_name] = rating
            raw_response_judge[query_id][model_name] = response
            score_models[rating].add(model_name)

        # Do pairwise-comparisons based on the absolute rating
        win_rates = {model_name: 0 for model_name in model_names}
        
        for score, models in score_models.items():
            model_list = list(models)
            random.shuffle(model_list)
            score_models[score] = model_list
        
        all_pairwise_prompts, all_model_combinations = [], []
        for score in score_models:
            if len(score_models[score]) > 1:
                for i in range(len(score_models[score])):
                    for j in range(i+1, len(score_models[score])):
                        model_i, model_j = score_models[score][i], score_models[score][j]
                        query_text = hf_queries[query_id]
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
        responses = openai_judge.batch_call(all_pairwise_prompts, 
                                            batch_size=args.batch_size, 
                                            temperature=args.temperature,
                                            name="pairwise",
                                            n=1)
        
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
        final_order = []
        for score in [5, 4, 3, 2, 1, 0]:
            if len(score_models[score]) > 1:
                # check the winrates
                winrates = {model_name: results_pairwise_winrate[query_id][model_name] for model_name in score_models[score]}
                winrates_sorted = sorted(winrates, key=lambda x: winrates[x], reverse=True)
                final_order.extend(winrates_sorted)
            elif len(score_models[score]) == 1:
                final_order.extend(score_models[score])
        
        final_sorted_results[query_id] = final_order
        
        os.makedirs(os.path.dirname(args.raw_output_scores), exist_ok=True)
        with open(args.raw_output_scores, "a", encoding="utf-8") as f:
            data = {
                "query_id": query_id,
                "ratings": results_judge.get(query_id, []),
                "outputs": raw_response_judge[query_id],
                "pairwise_winrates": results_pairwise_winrate.get(query_id, []),
                "pairwise_outputs": raw_response_pairwise_judge[query_id],
                "final_order": final_sorted_results[query_id]
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
    # # save the results in the original huggingface dataset
    # hf_documents = []
    
    # for query_id in hf_dataset:
    #     data = hf_dataset[query_id]
    #     final_order = final_sorted_results[query_id]
    #     data[f"{args.reference_model}-llm-judge"] = final_order
    #     hf_documents.append(data)
    
    # # save the results in the original huggingface dataset
    # hf_dataset_new = datasets.Dataset.from_pandas(pd.DataFrame(hf_documents))
    # print(f"Total documents in {args.language_code}: {len(hf_documents)}")
    # hf_dataset_new.push_to_hub(args.eval_hf_dataset, config_name=args.language_code, private=False, split=args.split)  

    # store results in a csv file
    # if not os.path.isfile(args.output_filepath):
    #     os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
    #     with open(args.output_filepath, "w", newline="\n") as f:
    #         writer = csv.writer(f, delimiter="\t")
    #         writer.writerow(["Model", "Language", "Reference", "Avg Mean", "Avg Stdev", "Temperatures"])

    # os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
    # with open(args.output_filepath, "a", newline="\n") as f:
    #     writer = csv.writer(f, delimiter="\t")
    #     for model_name in model_names:
    #         writer.writerow([model_name, 
    #                         args.language_code, 
    #                         args.reference_model,
    #                         round(avg_mean_results_judge[model_name], 3),
    #                         round(avg_stdev_results_judge[model_name], 3),
    #                         temperatures])
    #     f.write("\n\n")

    # save all ratings + llm_output separately in a jsonl file
    os.makedirs(os.path.dirname(args.raw_output_scores), exist_ok=True)
    with open(args.raw_output_scores, "w", encoding="utf-8") as f:
        for query_id in hf_dataset:
            data = {
                "query_id": query_id,
                "ratings": results_judge.get(query_id, []),
                "outputs": raw_response_judge[query_id],
                "pairwise_winrates": results_pairwise_winrate.get(query_id, []),
                "pairwise_outputs": raw_response_pairwise_judge[query_id]
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")