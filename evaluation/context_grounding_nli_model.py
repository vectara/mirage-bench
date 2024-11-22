import pandas as pd
import argparse
import re, csv, os
from typing import Dict, List
import stanza
from sentencex import segment

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch, datasets


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

class XNLIModel(AutoModelForSequenceClassification):
    def __init__(self, model_name: str, cache_dir: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True).to(self.device)
        self.label_names = ["entailment", "neutral", "contradiction"]

    def get_similarity_scores(self, premises: List[str], hypothesis: List[str], batch_size: int) -> Dict[str, float]:
        predictions = []
        with torch.no_grad():
            for itr in tqdm(range(0, len(premises), batch_size), desc=f"Computing NLI Scores with batch_size = {batch_size}..."):
                inputs = self.tokenizer(premises[itr:itr + batch_size], 
                                        hypothesis[itr:itr + batch_size], 
                                        max_length=self.tokenizer.model_max_length - 2, 
                                        padding=True,
                                        truncation=True, 
                                        return_tensors="pt"
                        )
                outputs = self.model(inputs["input_ids"].to(self.device), 
                                    attention_mask=inputs["attention_mask"].to(self.device))
                prediction = torch.softmax(outputs["logits"], -1).tolist()
                predictions += [{name: round(float(pred), 1) for pred, name in zip(preds, self.label_names)} for preds in prediction]
            
            return predictions


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
def sentence_tokenizer_with_citations(
        tokenizer: str, 
        rag_answer: str, 
        doc_ids: List[str],
        ) -> Dict[str, str]:
    
    final_sentences = []
    context_sentences = []
    context = re.search(r'Reason(.*?)Answer:', rag_answer, re.DOTALL)
    
    if context:
        # get the citations from the context
        context_string = context.group(1).strip().split("\n\n")[0].strip()
        context_sentences = tokenizer.sentence_tokenize(context_string)
    
    for context_sentence in context_sentences:
        sentence_citations = set()
        citations = re.findall(r"\[[^\]]*\]", context_sentence, re.DOTALL)
        
        if citations:
            for citation in citations:
                parsed_citation = citation.replace("[", "").replace("]", "")
                
                if "," in parsed_citation:
                    for cit in parsed_citation.split(","):
                        if cit in doc_ids:
                            context_sentence = context_sentence.replace(citation, "")
                            sentence_citations.add(cit)
                
                else:
                    if parsed_citation in doc_ids:
                        context_sentence = context_sentence.replace(citation, "")
                        sentence_citations.add(parsed_citation)
        
        if sentence_citations:
            final_sentences.append({"text": context_sentence, "citations": list(sentence_citations)})
    
    return final_sentences

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_hf_dataset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--language_code", type=str, required=True)
    parser.add_argument("--xnli_model", type=str, default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", required=False)
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    parser.add_argument("--cache_dir", type=str, required=False)
    parser.add_argument("--output_filepath", type=str, default="avg_context_grounding_nli_model_scores.tsv", required=False)
    args = parser.parse_args()

    # Loading XNLI model
    xnli_model = XNLIModel(args.xnli_model, cache_dir=args.cache_dir)
    load_dataset = datasets.load_dataset(args.eval_hf_dataset, args.language_code, split=args.split, cache_dir=args.cache_dir)

    hf_dataset = {}

    for row in load_dataset:
        hf_dataset[row["query_id"]] = row

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

    # Load the predictions and references
    documents = load_documents(hf_dataset)

    all_sentences = {query_id: {} for query_id in hf_dataset}

    for query_id in tqdm(hf_dataset, desc="Processing queries", total=len(hf_dataset)):
        for model_name in model_names:
            references = documents[query_id]

            for model_output in hf_dataset[query_id]["outputs"]:
                if model_output["model"] == model_name:
                    rag_output = model_output["output"]
                    predictions = sentence_tokenizer_with_citations(
                                    sentence_tokenizer, 
                                    rag_output,
                                    doc_ids=references)
                    all_sentences[query_id][model_name] = predictions

    # Computing grounding similarity
    entailment_scores_dict = {query_id: {} for query_id in hf_dataset}
    contradiction_scores_dict = {query_id: {} for query_id in hf_dataset}
    neutral_scores_dict = {query_id: {} for query_id in hf_dataset}

    avg_entailment_scores = {model_id: 0.0 for model_id in model_names}
    avg_contradiction_scores = {model_id: 0.0 for model_id in model_names}
    avg_neutral_scores = {model_id: 0.0 for model_id in model_names}
    
    for model_name in model_names:
        premises, hypothesis, counts = [], [], []
        all_query_ids = list(all_sentences.keys())

        print("\tEvaluating model: ", model_name)
        print("\tUsing model for XLNI eval: ", args.xnli_model)
        for query_id in tqdm(all_query_ids, total=len(all_query_ids), desc="Computing Grounding Similarity scores..."):
            count = 0
            
            for sentence_dict in all_sentences[query_id][model_name]:
                sentence, citations = sentence_dict["text"], sentence_dict["citations"]
                for doc_id in citations:
                    count += 1
                    premises.append(documents[query_id][doc_id])
                    hypothesis.append(sentence)
            counts.append(count)
    
        # get the similarity scores
        entailment_scores, contradiction_scores, neutral_scores = [], [], []
        predictions = xnli_model.get_similarity_scores(premises, hypothesis, batch_size=args.batch_size)

        start_idx = 0
        for query_id, count in zip(all_query_ids, counts):
            if count > 0:
                entailment_score, contradiction_score, neutral_score = 0, 0, 0
                for prediction in predictions[start_idx:start_idx+count]:
                    
                    entailment_score += prediction["entailment"]
                    contradiction_score += prediction["contradiction"]
                    neutral_score += prediction["neutral"]
            
                entailment_scores.append(entailment_score / count)
                contradiction_scores.append(contradiction_score / count)
                neutral_scores.append(neutral_score / count)
                
                # store the scores
                entailment_scores_dict[query_id][model_name] = entailment_score / count
                contradiction_scores_dict[query_id][model_name] = contradiction_score / count
                neutral_scores_dict[query_id][model_name] = neutral_score / count

                start_idx += count
    
        avg_entailment_scores[model_name] = sum(entailment_scores) / len(all_query_ids)
        avg_contradiction_scores[model_name] = sum(contradiction_scores) / len(all_query_ids)
        avg_neutral_scores[model_name] = sum(neutral_scores) / len(all_query_ids)
    
    # save the results in the original huggingface dataset
    hf_documents = []
    for query_id in hf_dataset:
        data = hf_dataset[query_id]
        data["xlni_context_entailment"] = entailment_scores_dict[query_id]
        data["xlni_context_neutral"] = neutral_scores_dict[query_id]
        data["xlni_context_contradiction"] = contradiction_scores_dict[query_id]
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
            writer.writerow(["Model", "Language", "XLNI Model", "Entailment", "Neutral", "Contradiction"])

    os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
    with open(args.output_filepath, "a", newline="\n") as f:
        writer = csv.writer(f, delimiter="\t")
        for model_name in model_names:
            writer.writerow([model_name, 
                            args.language_code, 
                            args.xnli_model,
                            round(avg_entailment_scores[model_name], 3),
                            round(avg_neutral_scores[model_name], 3),
                            round(avg_contradiction_scores[model_name], 3)])
        
        f.write("\n\n")
