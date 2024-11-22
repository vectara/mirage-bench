from openai import AzureOpenAI
import openai
import argparse, datasets
from typing import Dict, Tuple
from tqdm import tqdm
import re, os, csv
import numpy as np
import pandas as pd
import json

LLM_AS_A_JUDGE_PROMPT = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual. and/or cites relevant context.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual, and/or cites relevant context.
Score 3: The response is somewhat correct, accurate, and/or factual, and/or cites relevant context.
Score 4: The response is mostly correct, accurate, and factual, and cites relevant context.
Score 5: The response is completely correct, accurate, and factual and cites relevant context.

###Feedback:
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

    def chat(self, prompt: str, temperature: float, n: int):

        try:
            response = self.client.chat.completions.create(
                    model=self.deployment_name, # model = "deployment_name".
                    messages=[{"role": "user", "content": f"{prompt}"}],
                    temperature=temperature,
                    n=n,
            )
            output = response.choices[0].message.content
            return output
        
        except openai.BadRequestError as e:
            print(e)
            return ""

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

def postprocess_rating(output: str) -> Tuple[int, bool]:
    try:
        rating = int(re.search(r"\[RESULT\] (\d)", output).group(1))
        return rating, False
    except:
        return 0, True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_hf_dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--reference_model", type=str, required=True)
    parser.add_argument("--language_code", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None, required=False)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    parser.add_argument("--output_filepath", type=str, default="avg_openai_llm_as_a_judge_scores.tsv", required=False)
    parser.add_argument("--raw_output_scores", type=str, default="", required=False)
    parser.add_argument("--temperatures", type=str, nargs="+", default=[0.1], required=True)
    args = parser.parse_args()

    print(f"Loading Model: {args.model_name}")
    print(f"Loading Lang: {args.language_code}")
    temperatures = [float(x) for x in args.temperatures]
    # Load the predictions and references

    load_dataset = datasets.load_dataset(args.eval_hf_dataset, args.language_code, split=args.split, cache_dir=args.cache_dir)
    hf_dataset = {}
    
    for row in load_dataset:
        hf_dataset[row["query_id"]] = row
    
    print(f"\tLoading LLM Judge: {args.reference_model}")
    openai_judge = OpenAIClient(model=args.reference_model)

    model_names = []
    for output in list(hf_dataset.values())[0]["outputs"]:
        model_names.append(output["model"])
    
    results_judge = {query_id: {} for query_id in hf_dataset}
    raw_response_judge = {query_id: {} for query_id in hf_dataset}
    
    avg_mean_results_judge = {model_name: 0.0 for model_name in model_names}
    avg_stdev_results_judge = {model_name: 0.0 for model_name in model_names}

    for query_id in tqdm(hf_dataset, desc="Processing queries", total=len(hf_dataset)):
        # reference answer
        for model_output in hf_dataset[query_id]["outputs"]:
            if model_output["model"] == args.reference_model:
                reference_answer = parse_answer(model_output["output"])
                break
        
        for model_name in model_names:
            results_judge[query_id][model_name] = []
            raw_response_judge[query_id][model_name] = []
        
        # response to evaluate
        for model_name in model_names:
            for model_output in hf_dataset[query_id]["outputs"]:
                if model_output["model"] == model_name:
                    answer = parse_answer(model_output["output"])

                    prompt = LLM_AS_A_JUDGE_PROMPT.replace("{instruction}", hf_dataset[query_id]["prompt"])
                    prompt = prompt.replace("{response}", answer)
                    prompt = prompt.replace("{reference_answer}", reference_answer)

                    for temperature in temperatures:
                        response = openai_judge.chat(prompt, temperature, 1)
                        rating, error = postprocess_rating(response)
                        results_judge[query_id][model_name].append(rating)
                        raw_response_judge[query_id][model_name].append(response)

                    
                    # average rating
                    avg_mean_results_judge[model_name] += np.mean(results_judge[query_id][model_name])
                    avg_stdev_results_judge[model_name] += np.std(results_judge[query_id][model_name])
        
    # average of the llm judge ratings
    for model_name in model_names:
        avg_mean_results_judge[model_name] /= len(hf_dataset)
        avg_stdev_results_judge[model_name] /= len(hf_dataset)
    
    # save the results in the original huggingface dataset
    hf_documents = []
    for query_id in hf_dataset:
        data = hf_dataset[query_id]
        data[f"{args.reference_model}-judge"] = results_judge[query_id]
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
            writer.writerow(["Model", "Language", "Reference", "Avg Mean", "Avg Stdev", "Temperatures"])

    os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
    with open(args.output_filepath, "a", newline="\n") as f:
        writer = csv.writer(f, delimiter="\t")
        for model_name in model_names:
            writer.writerow([model_name, 
                            args.language_code, 
                            args.reference_model,
                            round(avg_mean_results_judge[model_name], 3),
                            round(avg_stdev_results_judge[model_name], 3),
                            temperatures])
        f.write("\n\n")

    # save all ratings + llm_output separately in a jsonl file
    os.makedirs(os.path.dirname(args.raw_output_scores), exist_ok=True)
    with open(args.raw_output_scores, "w", encoding="utf-8") as f:
        for query_id in raw_response_judge:
            data = {
                "query_id": query_id,
                "ratings": results_judge.get(query_id, []),
                "outputs": raw_response_judge[query_id]
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")