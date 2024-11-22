import evaluate, datasets
import pandas as pd
from tqdm import tqdm
import argparse
import re, csv, os
from typing import Dict
from transformers import AutoTokenizer, T5Tokenizer
from rouge_score import rouge_scorer

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

WITH_STEMMER = ["ar", "bn", "hi", "en", "fi", "fr", "de", "ru", "es"]
WITHOUT_STEMMER = ["zh", "th", "ja"]

# Load the predictions and references
def parse_answer(rag_answer: str) -> Dict[str, str]:
    answer = re.search(r'Answer:(.*?)$', rag_answer, re.DOTALL)
    
    if answer:
        answer_string = answer.group(1).strip().split("\n\n")[0].strip()
        
        # remove the citations from the answer
        new_answer = re.findall(r"\[[^\]]*\]", answer_string, re.DOTALL)

        if new_answer:
            for citation in new_answer:
                answer_string = answer_string.replace(citation, "")
    
    return answer_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_hf_dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--reference_model", type=str, required=True)
    parser.add_argument("--language_code", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=False)
    parser.add_argument("--output_filepath", type=str, default="avg_rouge_and_blue_scores.tsv", required=False)
    args = parser.parse_args()

    print(f"Loading Reference: {args.reference_model}")
    print(f"Loading Lang: {args.language_code}")

    sacrebleu = evaluate.load("sacrebleu")

    use_stemmer = True if args.language_code in WITH_STEMMER else False
    rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer, lang=ISO_TO_LANG[args.language_code])

    load_dataset = datasets.load_dataset(args.eval_hf_dataset, args.language_code, split=args.split, cache_dir=args.cache_dir)
    hf_dataset = {}
    
    for row in load_dataset:
        hf_dataset[row["query_id"]] = row
    
    model_names = []
    for output in list(hf_dataset.values())[0]["outputs"]:
        model_names.append(output["model"])

    results_bleu = {query_id: {} for query_id in hf_dataset}
    results_rouge = {query_id: {} for query_id in hf_dataset}

    avg_results_bleu = {model_id: 0 for model_id in model_names}
    avg_results_rouge = {model_id: 0 for model_id in model_names}
    
    for query_id in tqdm(hf_dataset, desc="Processing queries", total=len(hf_dataset)):
        # reference answer
        for model_output in hf_dataset[query_id]["outputs"]:
            if model_output["model"] == args.reference_model:
                reference_answer = parse_answer(model_output["output"])
                break
        
        for model_name in model_names:
            for model_output in hf_dataset[query_id]["outputs"]:
                if model_output["model"] == model_name:
                    answer = parse_answer(model_output["output"])
                    sacrebleu_score = sacrebleu.compute(predictions=[answer], references=[reference_answer])
                    rouge_score = rouge_scorer.score(answer, reference_answer)

                    # store the result
                    avg_results_bleu[model_name] += sacrebleu_score["score"] / 100
                    avg_results_rouge[model_name] += rouge_score["rougeL"].fmeasure

                    # save the scores
                    results_bleu[query_id][model_name] = sacrebleu_score["score"] / 100
                    results_rouge[query_id][model_name] = rouge_score["rougeL"].fmeasure
        
    
    # average of the bleu and rouge results
    for model_name in model_names:
        avg_results_bleu[model_name] /= len(hf_dataset)
        avg_results_rouge[model_name] /= len(hf_dataset)

    # save the results in the original huggingface dataset
    hf_documents = []
    for query_id in hf_dataset:
        data = hf_dataset[query_id]
        data["answer_bleu"] = results_bleu[query_id]
        data["answer_rougeL"] = results_rouge[query_id]
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
            writer.writerow(["Model", "Language", "Reference", "BLEU", "MUL ROUGE-L"])

    os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
    with open(args.output_filepath, "a", newline="\n") as f:
        writer = csv.writer(f, delimiter="\t")
        for model_name in model_names:
            writer.writerow([model_name, 
                            args.language_code, 
                            args.reference_model,
                            round(avg_results_bleu[model_name], 3),
                            round(avg_results_rouge[model_name], 3)])
        f.write("\n\n")
