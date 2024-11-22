import choix
import json, os
import random
import numpy as np
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from openai import AzureOpenAI
from itertools import combinations
import openai, datasets, argparse

# load the prompts
from bootstrap_prompts import ChainOfThoughtPrompt

seed = 42
random.seed(seed)

def load_jsonl(file_paths):
    with open(file_paths[0], 'r') as file1:
        with open(file_paths[1], 'r') as file2:
            results = {json.loads(line)["query_id"]: json.loads(line) for line in file1}
            fluency = {json.loads(line)["query_id"]: json.loads(line)["ratings"] for line in file2}
            
            for query_id in results:
                results[query_id]["ratings"] = fluency[query_id]
    return results

RESULTS_DIR = "/u3/n3thakur/projects/vectara/vectara-translation/datasets"
FLUENCY_DIR = "/u3/n3thakur/projects/vectara/evaluation/scores/may-10th/fluency"

# LANGUAGES = ["ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "yo", "zh"]
LANGUAGES = ["en"]

MODEL_NAMES = ["cohereforai-c4ai-command-r-plus", 
                "gpt-3.5-turbo-azure", 
                "gpt-4-azure", 
                "meta-llama/Meta-Llama-3-70B-Instruct", 
                "meta-llama/Meta-Llama-3-8B-Instruct", 
                "mistralai/Mistral-7B-Instruct-v0.2"]

SHORT_MODEL_NAMES = [
    'cmd-r', 'gpt-3.5', 'gpt-4', 'llama-70', 'llama-8', 'mist-7b'
]

FILES = {}
for language in LANGUAGES:
    FILES[language] = []
    for model_name in MODEL_NAMES:
        FILES[language].append(
        [f"{RESULTS_DIR}/{model_name}/miracl-eval/{language}-miracl-raft-eval-dev-small-0-100.jsonl", 
         f"{FLUENCY_DIR}/{model_name}/llama-3-8b/{language}-miracl-raft-eval-dev-small-0-100-raw-output.jsonl"],
        )

class OpenAIClient:
    def __init__(self, 
                 model: str = None, 
                 endpoint: str = None, 
                 api_key: str = None, 
                 api_version: str = "2024-02-01", 
                 wait: int = 60):
        
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


class LLMJudge:
    def __init__(self, judge_name: str, model_names: List[str], files: List[List[str]], prompt_template: str):

        self.datasets = [
            (model_names[i], load_jsonl(files[i])) for i in range(len(model_names))
        ]
        self.dataset_idx = [i for i in range(len(self.datasets))]
        self.model_names = model_names
        self.question_ids = self.get_all_questions()
        # self.client = OpenAIClient(judge_name)
        self.prompt_template = prompt_template

        print(f"Loaded data for {len(self.model_names)} models. Found {len(self.question_ids)} unique queries.")
    
    def get_all_questions(self):
        """Get all the unique questions from the datasets."""
        question_ids = set()

        for _, v in self.datasets:
            for query_id in v.keys():
                question_ids.add(query_id)
        
        return question_ids

    def choose_models(self):
        """Randomly choose two models."""
        return random.sample(self.dataset_idx, 2)
    
    def choose_all_models(self):
        """Randomly choose two models."""
        return list(combinations(self.dataset_idx, 2))

    def choose_question(self):
        """Randomly choose a question id."""
        return random.choice(list(self.question_ids))
        
    def fluency(self, model_idx, question_id):
        """Sample the fluency of the given model on the given question."""
        # print(self.datasets[model_idx][1]["outputs"][], "\n\n")
        # model_output = self.datasets[model_idx][1][question_id]["outputs"][self.model_names[model_idx]]
        if question_id not in self.datasets[model_idx][1]:
            return None
        if "ratings" not in self.datasets[model_idx][1][question_id]:
            return None
        return random.choice(list(self.datasets[model_idx][1][question_id]["ratings"]))


def simulate_tournament(num_matches=100_000, judge=None):
    """Simulate a tournament with the specified number of head-to-head matchups."""
    preferences, sampled_queries = [], []
    
    for i in range(num_matches):
        # choose a question and add it to the list of sampled queries
        question = judge.choose_question()
        sampled_queries.append(question)
        
        # choose all possible nc2 pairs of models
        for model_a, model_b in judge.choose_all_models():
            score_a, score_b = judge.fluency(model_a, question), judge.fluency(model_b, question)
            if score_a is None or score_b is None:
                continue

            # The scoring function here is really simple. We just use
            # the fluency to judge which model produces the best output.
            # However, in reality we would use an LLM judge here.
            if score_a > score_b:
                preferences.append((model_a, model_b))
            elif score_b > score_a:
                preferences.append((model_b, model_a))
            #print(f"[{question}] {model_a} vs. {model_b} = {score_a} vs {score_b}")
    
    try:
        result = choix.ilsr_pairwise(len(judge.dataset_idx), preferences)
    
    except (ValueError, RuntimeError) as e:
        result = choix.opt_pairwise(len(judge.dataset_idx), preferences)

    return result, sampled_queries

def print_results(tournament_count, mean_scores, median_scores, ci_95, model_names):
    print("===================================")
    print("B E S T   M O D E L S   R A N K E D")
    print("===================================")
    print()
    print(f"Num Tournaments: {tournament_count}")
    print()
    print("   MODEL NAME                                MEAN          95% CI")
    print("   ==============================================================")
    for i, model_idx in enumerate(reversed(np.argsort(mean_scores))):
        minus = ci_95[0][model_idx] - mean_scores[model_idx]
        plus = ci_95[1][model_idx] - mean_scores[model_idx]
        print(f"{i + 1}. {model_names[model_idx]:<40} {mean_scores[model_idx]: 4.3f}     {minus: 3.2f}/+{plus:3.2f}")

def p_win(pair, params):
    # Placeholder for the actual choix.probabilities function
    # Assume it returns a tuple (prob_i, prob_j) for the models in 'pair'
    i, j = pair
    return choix.probabilities([i, j], params)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_hf_dataset", type=str, required=False, default="nthakur/miracl-raft-eval")
    parser.add_argument("--split", type=str, required=False, default="dev.small")
    parser.add_argument("--cache_dir", type=str, required=False, default="/u3/n3thakur/projects/cache")
    parser.add_argument("--output_filepath", type=str, default="avg_citation_map_recall_scores.tsv", required=False)
    args = parser.parse_args()

    FEATURES = [
        "language_detection", 
        "citation_MAP@6",
        "citation_Recall@6",
        "xlni_context_entailment",
        "answer_bleu",
        "answer_rougeL",
    ]

    
    for lang in LANGUAGES:
        print(f"Loading feature dataset for Language: {lang} ....")
        load_dataset = datasets.load_dataset(args.eval_hf_dataset, lang, split=args.split, cache_dir=args.cache_dir)

        # Load the predictions and references
        hf_dataset = {}
        
        for row in load_dataset:
            hf_dataset[row["query_id"]] = row

        print(f"Evaluating Language: {lang} ....")
        judge = LLMJudge("gpt4o", MODEL_NAMES, FILES[lang], ChainOfThoughtPrompt())

        num_models = len(MODEL_NAMES)
        
        num_tournaments = 200
        matches_per_tournament = 50

        all_scores = np.zeros((num_tournaments, num_models))
        train_X = np.zeros((num_tournaments * num_models, len(FEATURES)))
        train_Y = np.zeros((num_tournaments * num_models, 1))

        for tournament_counter in range(num_tournaments):
            params, queries = simulate_tournament(num_matches=matches_per_tournament, judge=judge)
            feature_dict = {model_name: {} for model_name in MODEL_NAMES}

            
            for query_id in queries:
                hf_dataset[query_id]["features"] = {}
                for feature_name in FEATURES:
                    for model_name, feature_value in hf_dataset[query_id][feature_name].items():
                        if feature_value is not None and model_name in MODEL_NAMES:
                            if feature_name not in feature_dict[model_name]:
                                feature_dict[model_name][feature_name] = 0.0
                            feature_dict[model_name][feature_name] += feature_value
            
            all_scores[tournament_counter] = params
            
            for model_name in MODEL_NAMES:
                for feature_name in FEATURES:
                    if feature_name in feature_dict[model_name]:
                        feature_dict[model_name][feature_name] /= len(queries)
            
            for idx, model_name in enumerate(MODEL_NAMES):
                train_X[tournament_counter * num_models + idx] = [feature_dict[model_name][feature_name] for feature_name in FEATURES]
                train_Y[tournament_counter * num_models + idx] = params[idx]

        mean_scores = np.mean(all_scores, axis=0)
        median_scores = np.median(all_scores, axis=0)
        ci_95 = np.percentile(all_scores, [2.5, 97.5], axis=0)

        # Train Linear regression model
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=seed)
        reg = LinearRegression().fit(X_train, y_train)
        print(f"Training Score: {reg.score(X_train, y_train)}")
        print(f"Test Score: {reg.score(X_test, y_test)}")
        print(f"Model Coefficients: {reg.coef_}")
        print(f"Features: {FEATURES}")

        MODEL_NAMES_NEW = MODEL_NAMES + ['google/gemma-7b-it'] 

        # Final ranking prediction using the model
        final_scores = {model_name: {} for model_name in MODEL_NAMES_NEW}
        
        for model_name in MODEL_NAMES_NEW:
            for query_id in hf_dataset:
                all_features = []
                for feature_name in FEATURES:
                    feature_value = hf_dataset[query_id][feature_name][model_name]
                    if feature_value is not None:
                        all_features.append(feature_value)
                    else:
                        all_features.append(0.0)
                
                final_scores[model_name][query_id] = reg.predict(np.array(all_features).reshape(1, -1))[0][0]
        
        # Rank the models based on the final scores
        final_mean = {model_name: np.mean(list(final_scores[model_name].values())) for model_name in MODEL_NAMES_NEW}
        final_ci_95 = {model_name: np.percentile(list(final_scores[model_name].values()), [2.5, 97.5]) for model_name in MODEL_NAMES_NEW}
        final_scores = {k: v for k, v in sorted(final_mean.items(), key=lambda item: item[1], reverse=True)}

        
        print("=========================================")
        print("F I N A L   L E A R N E D   R A N K I N G")
        print("=========================================")
        print()
        print("   MODEL NAME                                MEAN         95% CI")
        print("   ==============================================================")
        for i, (model_name, score) in enumerate(final_scores.items()):
            print(f"{i + 1}. {model_name:<40} {score: 4.3f}    {final_ci_95[model_name][0]: 3.2f}/+{final_ci_95[model_name][1]:3.2f}")
        print("\n\n")

        # # Finally, for the sake of producing a heatmap, run one more tournament.
        # params = simulate_tournament(num_matches=matches_per_tournament, judge=judge)
        # prob_matrix = np.zeros((num_models, num_models))

        # # Compute probabilities
        # for i in range(num_models):
        #     for j in range(num_models):
        #         if i != j:
        #             prob_i, prob_j = p_win([i, j], params=params)
        #             prob_matrix[i, j] = prob_i
        
        print_results(num_tournaments, mean_scores, median_scores, ci_95, MODEL_NAMES)
        print()

        # Show distribution of all scores
        boxplot_data = []
        for i in range(num_models):
            for score in all_scores[:, i]:
                boxplot_data.append({'Model': f'{SHORT_MODEL_NAMES[i]}', 'Score': score})

        # Convert to a DataFrame
        df = pd.DataFrame(boxplot_data)

        # Create the box plot
        plt.figure(figsize=(5, 4))
        sns.boxplot(x='Model', y='Score', data=df)
        plt.title('Distribution of Logits for Each Model')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.show()

        # # Create a heatmap
        # plt.figure(figsize=(5, 4))
        # sns.heatmap(prob_matrix, annot=True, cmap='viridis', xticklabels=MODEL_NAMES, yticklabels=MODEL_NAMES)
        # plt.title('Probabilities of Model A beating Model B')
        # plt.xlabel('Model B')
        # plt.ylabel('Model A')
        # plt.show()
