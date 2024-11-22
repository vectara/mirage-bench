import datasets
import json
import argparse
import re, csv, os
from typing import Dict, List
import stanza
from sentencex import segment
import pandas as pd

from FlagEmbedding import FlagReranker
from tqdm import tqdm

class StanzaTokenizer:
    def __init__(self, language_code: str):
        self.nlp = stanza.Pipeline(lang=language_code, processors='tokenize')

    def sentence_tokenize(self, text):
        doc = self.nlp(text)
        return [sentence.text for sentence in doc.sentences]

class SentenceXTokenizer:
    def __init__(self, language_code: str):
        self.language_code = language_code
    
    def sentence_tokenize(self, text):
        return list(segment(self.language_code, text))

class Reranker:
    def __init__(self, model_name: str, cache_dir: str = None, use_fp16: bool = True):
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16, cache_dir=cache_dir)
    
    def compute_score(self, pairs: List[List[str]], batch_size: int, name: str = None) -> List[float]:
        predictions = []
        for itr in tqdm(range(0, len(pairs), batch_size), desc=f"Computing Reranker scores for {name} with batch_size = {batch_size}..."):
            end_pointer = len(pairs) if itr + batch_size > len(pairs) else itr + batch_size
            scores = self.reranker.compute_score(pairs[itr:end_pointer])
            if not isinstance(scores, list): scores = [scores]
            predictions += scores
        return predictions

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


def load_gold_documents(filepath: str) -> Dict[str, str]:
    answer_dict = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            answer_dict[data["query_id"]] = {}
            prompt = data["prompt"]
            context = prompt.split("\n\nContext:")[1].split("\n\nInstruction")[0].strip()
            documents = context.split("\n")
            for document in documents:
                doc_id = document.split("]")[0].replace("[", "").strip()
                doc_text = "]".join(document.split("]")[1:]).strip()
                answer_dict[data["query_id"]][doc_id] = doc_text

    return answer_dict

# Load the predictions and references
def find_citations(
    rag_answer: str, 
    doc_ids: List[str]
    ) -> Dict[str, str]:

    sentence_citations = set()
    
    if rag_answer:
        citations = re.findall(r"\[[^\]]*\]", rag_answer, re.DOTALL)

        if citations:
            for citation in citations:
                parsed_citation = citation.replace("[", "").replace("]", "")
                
                if "," in parsed_citation:
                    for cit in parsed_citation.split(","):
                        if cit in doc_ids:
                            rag_answer = rag_answer.replace(citation, "")
                            sentence_citations.add(cit)
                
                else:
                    if parsed_citation in doc_ids:
                        rag_answer = rag_answer.replace(citation, "")
                        sentence_citations.add(parsed_citation)
    
    return list(sentence_citations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_hf_dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--language_code", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=False)
    parser.add_argument("--output_filepath", type=str, default="avg_query_cited_passages_reranker.tsv", required=False)
    parser.add_argument("--reranker_model", type=str, default='BAAI/bge-reranker-v2-m3', required=False)
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    args = parser.parse_args()

    print(f"Language: {args.language_code}")
    load_dataset = datasets.load_dataset(args.eval_hf_dataset, args.language_code, split=args.split, cache_dir=args.cache_dir)
    hf_dataset = {}

    # Load the HF dataset into a dictionary
    for row in load_dataset:
        hf_dataset[row["query_id"]] = row

    # load queries
    queries = load_queries(hf_dataset)
    documents = load_documents(hf_dataset)

    # Start the evaluation
    model_names = []
    for output in list(hf_dataset.values())[0]["outputs"]:
        model_names.append(output["model"])

    # Stanza Tokenizer
    try:
        print("\tUsing Stanza Tokenizer...")
        sentence_tokenizer = StanzaTokenizer(language_code=args.language_code)
    
    except Exception as e:
        print("\tStanza Tokenizer not available, using SentenceX Tokenizer...")
        sentence_tokenizer = SentenceXTokenizer(language_code=args.language_code)
    
    all_sentences = {query_id: {model_name: [] for model_name in model_names} for query_id in hf_dataset}
    
    # Load the predictions and references
    for query_id in tqdm(hf_dataset, desc="Processing queries", total=len(hf_dataset)):
        for model_name in model_names:
            references = list(documents[query_id].keys())

            for model_output in hf_dataset[query_id]["outputs"]:
                if model_output["model"] == model_name:
                    rag_output = model_output["output"]
                    
                    predictions = find_citations(
                        rag_output, 
                        doc_ids=references)
                    
                    all_sentences[query_id][model_name] = predictions
    
    # computing the similarity scores
    reranker_scores = {query_id: {} for query_id in hf_dataset}
    avg_reranker_scores = {model_id: 0.0 for model_id in model_names}
    
    print("\tUsing model for reranker eval: ", args.reranker_model)
    reranker = Reranker(args.reranker_model, use_fp16=True)

    for model_name in model_names:
        scores, final_scores, counts = [], [], []
        
        for query_id in all_sentences:
            count = 0
            for doc_id in all_sentences[query_id][model_name]:
                scores.append([queries[query_id], documents[query_id][doc_id]])
                count += 1
            counts.append(count)

        # compute all the reranker scores for model_name
        reranker_predictions = reranker.compute_score(scores, batch_size=args.batch_size, name=model_name)
        
        start_idx = 0
        
        for query_id, count in zip(all_sentences, counts):
            if count > 0:
                reranker_score = 0
                for score in reranker_predictions[start_idx:start_idx+count]:
                    reranker_score += score
                
                reranker_scores[query_id][model_name] = reranker_score / count
                final_scores.append(reranker_score / count)
                start_idx += count

        avg_reranker_scores[model_name] = sum(final_scores) / len(all_sentences)

    # save the results in the original huggingface dataset
    hf_documents = []
    for query_id in hf_dataset:
        data = hf_dataset[query_id]
        reranker_name = f"{args.reranker_model.split('/')[1]}-score"
        data[reranker_name] = reranker_scores[query_id]
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
            writer.writerow(["Model", "Language", "Reranker Model", "Cited Passage Score"])

    os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
    with open(args.output_filepath, "a", newline="\n") as f:
        writer = csv.writer(f, delimiter="\t")
        for model_name in model_names:
            writer.writerow([model_name, 
                            args.language_code, 
                            args.reranker_model,
                            round(avg_reranker_scores[model_name], 3)])
        
        f.write("\n\n")
