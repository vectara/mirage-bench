from langdetect import detect_langs
from tqdm import tqdm
from lingua import Language, LanguageDetectorBuilder
import pandas as pd

import json
import argparse
import re, csv, os
from typing import Dict, List
import datasets

## Only for yoruba language
languages = [Language.ENGLISH, Language.YORUBA]
yoruba_detector = LanguageDetectorBuilder.from_languages(*languages).build()

LANG_TO_CODE = {
    "YORUBA": "yo",
    "ENGLISH": "en",
}


def text_postprocessing(text: str, filter_ids: str, reason: str = "##Reason:", answer: str = "##Answer:") -> str:
    for filter_id in filter_ids:
        text = text.replace(filter_id, "")
    text = text.replace(reason, "").replace(answer, "").replace("[", "").replace("]", "").replace("\n", " ").strip()
    text = " ".join(text.split())
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_hf_dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--language_code", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=False)
    parser.add_argument("--output_filepath", type=str, default="avg_citation_and_language.tsv", required=False)
    args = parser.parse_args()

    print(f"Loading Lang: {args.language_code}")

    load_dataset = datasets.load_dataset(args.eval_hf_dataset, args.language_code, split=args.split, cache_dir=args.cache_dir)
    hf_dataset = {}
    
    for row in load_dataset:
        hf_dataset[row["query_id"]] = row
    
    model_names = []
    for output in list(hf_dataset.values())[0]["outputs"]:
        model_names.append(output["model"])

    results_lang = {query_id: {} for query_id in hf_dataset}
    results_en = {query_id: {} for query_id in hf_dataset}
    results_other = {query_id: {} for query_id in hf_dataset}


    
    for query_id in tqdm(hf_dataset, desc="Processing queries", total=len(hf_dataset)):
        for model_name in model_names:
            # initialization
            results_lang[query_id][model_name] = 0.0
            results_en[query_id][model_name] = 0.0
            results_other[query_id][model_name] = 0.0
            doc_ids = hf_dataset[query_id]["positive_ids"] + hf_dataset[query_id]["negative_ids"]

            for model_output in hf_dataset[query_id]["outputs"]:
                if model_output["model"] == model_name:
                    rag_output = text_postprocessing(model_output["output"], filter_ids=doc_ids)
                    detected_lang = {}
                    try:
                        # langdetect for all languages except yoruba (yo)
                        # https://github.com/Mimino666/langdetect
                        if args.language_code != "yo":
                            for lang_object in detect_langs(rag_output):
                                if lang_object.lang == "zh-cn": # convert to zh
                                    detected_lang["zh"] = lang_object.prob
                                else:
                                    detected_lang[lang_object.lang] = lang_object.prob
                        
                        # for yoruba language (using Lingua-py)
                        # https://github.com/pemistahl/lingua-py
                        elif args.language_code == "yo":
                            confidence_values = yoruba_detector.compute_language_confidence_values(rag_output)
                            for confidence in confidence_values:
                                detected_lang[LANG_TO_CODE[confidence.language.name]] = confidence.value

                    except Exception as e:
                        print(f"Error in detecting language for query_id: {query_id} and model: {model_name}")
            
            for lang in detected_lang:
                if args.language_code == lang:
                    results_lang[query_id][model_name] = detected_lang[args.language_code]
                elif "en" == lang:
                    results_en[query_id][model_name] = detected_lang["en"]
                else:
                    results_other[query_id][model_name] += detected_lang[lang]
    
    # save the results in the original huggingface dataset
    hf_documents = []
    for query_id in hf_dataset:
        data = hf_dataset[query_id]
        data["language_detection"] = results_lang[query_id]
        data["en_detection"] = results_en[query_id]
        data["other_detection"] = results_other[query_id]
        hf_documents.append(data)
    
    # save the results in the original huggingface dataset
    hf_dataset_new = datasets.Dataset.from_pandas(pd.DataFrame(hf_documents))
    print(f"Total documents in {args.language_code}: {len(hf_documents)}")
    hf_dataset_new.push_to_hub(args.eval_hf_dataset, config_name=args.language_code, private=False, split=args.split)    


    # Computing avg language detection scores
    avg_lang_detection = {model_name: 0.0 for model_name in model_names}
    avg_en_detection = {model_name: 0.0 for model_name in model_names}
    avg_other_detection = {model_name: 0.0 for model_name in model_names}

    for query_id in hf_dataset:
        for model_name in model_names:
            avg_lang_detection[model_name] += results_lang[query_id][model_name]
            avg_en_detection[model_name] += results_en[query_id][model_name]
            avg_other_detection[model_name] += results_other[query_id][model_name]
    
    avg_lang_detection = {model_name: score / len(hf_dataset) for model_name, score in avg_lang_detection.items()}
    avg_en_detection = {model_name: score / len(hf_dataset) for model_name, score in avg_en_detection.items()}
    avg_other_detection = {model_name: score / len(hf_dataset) for model_name, score in avg_other_detection.items()}


    # store results in a csv file
    if not os.path.isfile(args.output_filepath):
        os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
        with open(args.output_filepath, "w", newline="\n") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["Model", "Language", "Avg. Lang Detection", "Avg. EN Detection", "Avg. Other Detection"])

    os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
    with open(args.output_filepath, "a", newline="\n") as f:
        writer = csv.writer(f, delimiter="\t")
        for model_name in model_names:
            writer.writerow([model_name, 
                            args.language_code, 
                            round(avg_lang_detection[model_name], 3),
                            round(avg_en_detection[model_name], 3),
                            round(avg_other_detection[model_name], 3),
                            ])
        
        f.write("\n\n")
        
