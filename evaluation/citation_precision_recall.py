import pandas as pd
import argparse
import re, csv, os
import pytrec_eval
import datasets
from typing import Dict, List

def load_gold_predictions(hf_dataset: str) -> Dict[str, str]:
    qrels = {}

    for query_id in hf_dataset:

        if query_id not in qrels:
            qrels[query_id] = {}

        for doc_id in hf_dataset[query_id]["positive_ids"]:
            qrels[query_id][doc_id] = 1
        
        for doc_id in hf_dataset[query_id]["negative_ids"]:
            qrels[query_id][doc_id] = 0

    return qrels

# Load the predictions and references
def parse_citations(rag_answer: str, filter_set: List[str]) -> Dict[str, str]:
    runfile, final_citations = {}, []

    context = re.search(r'Reason(.*?)Answer:', rag_answer, re.DOTALL)
    answer = re.search(r'Answer:(.*?)$', rag_answer, re.DOTALL)
    
    if context:
        # get the citations from the context
        context_string = context.group(1).strip().split("\n\n")[0].strip()
        context_citations = re.findall(r"\[[^\]]*\]", context_string, re.DOTALL)
    
    if answer:
        # get the citations from the answer
        answer_string = answer.group(1).strip().split("\n\n")[0].strip()
        answer_citations = re.findall(r"\[[^\]]*\]", answer_string, re.DOTALL)

    generated_citations = context_citations + answer_citations
    
    if generated_citations:
        for citation in generated_citations:
            citation = citation.replace("[", "").replace("]", "")
            if "," in citation:
                final_citations += [cit for cit in citation.split(",") if cit in filter_set]
            
            elif citation in filter_set:
                final_citations.append(citation)
    
    if final_citations:
        for idx, citation in enumerate(final_citations):
            runfile[citation] = len(final_citations) - idx

    return runfile

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_hf_dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--language_code", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=False)
    parser.add_argument("--k_values", type=str, nargs="+", default=[10], required=False)
    parser.add_argument("--output_filepath", type=str, default="avg_citation_map_recall_scores.tsv", required=False)
    args = parser.parse_args()

    print(f"Loading Lang: {args.language_code}")

    load_dataset = datasets.load_dataset(args.eval_hf_dataset, args.language_code, split=args.split, cache_dir=args.cache_dir)
    hf_dataset = {}
    
    for row in load_dataset:
        hf_dataset[row["query_id"]] = row
    
    model_names = []
    for output in list(hf_dataset.values())[0]["outputs"]:
        model_names.append(output["model"])

    qrels = load_gold_predictions(hf_dataset)
    runfile = {model_name: {} for model_name in model_names}
    
    for query_id in hf_dataset:
        for model_name in model_names:
            for model_output in hf_dataset[query_id]["outputs"]:
                if model_output["model"] == model_name:
                    runfile[model_name][query_id] = parse_citations(model_output["output"], filter_set=qrels[query_id])
    
    map_results = {k: {query_id: {} for query_id in hf_dataset} for k in args.k_values}
    recall_results = {k: {query_id: {} for query_id in hf_dataset} for k in args.k_values}
    map_avg_results = {k: {model_name: 0.0 for model_name in runfile} for k in args.k_values}
    recall_avg_results = {k: {model_name: 0.0 for model_name in runfile} for k in args.k_values}
    
    
    map_string = "map_cut." + ",".join([str(k) for k in args.k_values])
    recall_string = "recall." + ",".join([str(k) for k in args.k_values])

    for model_name in runfile:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, recall_string})
        scores = evaluator.evaluate(runfile[model_name])

        for query_id in scores:
            for k in args.k_values:
                map_results[k][query_id][model_name] = scores[query_id]["map_cut_" + str(k)]
                recall_results[k][query_id][model_name] = scores[query_id]["recall_" + str(k)]

                map_avg_results[k][model_name] += scores[query_id]["map_cut_" + str(k)]
                recall_avg_results[k][model_name] += scores[query_id]["recall_" + str(k)]
    
    # Computing mAP@K
    for k in args.k_values:
        for model_name in model_names:
            map_avg_results[k][model_name] = round(map_avg_results[k][model_name] / len(hf_dataset), 3)
            recall_avg_results[k][model_name] = round(recall_avg_results[k][model_name] / len(hf_dataset), 3)
    
    # save the results in the original huggingface dataset
    hf_documents = []
    for query_id in hf_dataset:
        for k_value in args.k_values:
            data = hf_dataset[query_id]
            data[f"citation_MAP@{k}"] = map_results[k][query_id]
            data[f"citation_Recall@{k}"] = recall_results[k][query_id]
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
            header = ["Model", "Language"]
            for k in args.k_values:
                header += [f"MAP@{k}", f"Recall@{k}"]
            writer.writerow(header)

    os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
    with open(args.output_filepath, "a", newline="\n") as f:
        writer = csv.writer(f, delimiter="\t")
        for model_name in model_names:
            row = [model_name, args.language_code]
            for k in args.k_values:
                row += [map_avg_results[k][model_name], recall_avg_results[k][model_name]]
            writer.writerow(row)

        f.write("\n\n")

