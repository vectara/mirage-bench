"""
Usage: python weight_learning_bootstrap_elo_judge.py \
    --eval_hf_dataset nthakur/mirage-eval-rag-output \
    --llm4judge_scores /u3/n3thakur/projects/vectara/evaluation/scores/june-25th/raw/llm4judge-backup/pairwise_gpt4o_llm_as_a_judge_ja_scores.jsonl \
    --split dev --language ja \
    --regression_model_name RandomForestRegressor \
    --cache_dir /u3/n3thakur/projects/cache \
    --output_filepath avg_citation_map_recall_scores.tsv

"""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import choix
import json, os
import random
import numpy as np
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from itertools import combinations
import datasets, argparse

seed = 42
random.seed(seed)

def load_json_file(filepath):
    results = {}
    with open(filepath, 'r') as file:
        for line in file:
            row = json.loads(line)
            query_id = row["query_id"]
            results[query_id] = row["pairwise_winrates"]
    return results

# RESULTS_DIR = "/u3/n3thakur/projects/vectara/vectara-translation/datasets"
# FLUENCY_DIR = "/u3/n3thakur/projects/vectara/evaluation/scores/may-10th/fluency"

# LANGUAGES = ["ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "yo", "zh"]

MODEL_NAMES = [
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "cohereforai-c4ai-aya-23",
    "cohereforai-c4ai-command-r",
    "cohereforai-c4ai-command-r-plus",
    "google/gemma-1.1-2b-it",
    "google/gemma-1.1-7b-it",
    "gpt-3.5-turbo-azure",
    "gpt-4-azure",
    "gpt-4o-azure",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/Phi-3-medium-128k-instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3-small-8k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

class RegressionModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if self.model_name == "LinearRegression":
            self.model = LinearRegression()
        elif self.model_name == "RandomForestRegressor":
            self.model = RandomForestRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
    



class LLMJudge:
    def __init__(self, model_names: List[str], llm4judge_result: List[List[str]]):

        self.pairwise_results = load_json_file(llm4judge_result)
        self.dataset_idx = [i for i in range(len(model_names))]
        self.model_names = model_names
        self.question_ids = self.get_all_questions()

        print(f"Loaded data for {len(self.model_names)} models. Found {len(self.question_ids)} unique queries.")
    
    def get_all_questions(self):
        """Get all the unique questions from the datasets."""
        question_ids = set()

        for key in self.pairwise_results.keys():
            question_ids.add(key)
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
        
    def winrate(self, model_idx, question_id):
        """Sample the winrate of the given model on the given question."""
        if question_id not in self.pairwise_results:
            return None
        return self.pairwise_results[question_id][self.model_names[model_idx]]


def simulate_tournament(num_matches=100_000, judge=None):
    """Simulate a tournament with the specified number of head-to-head matchups."""
    preferences, sampled_queries = [], []
    
    for idx in range(num_matches):
        # choose a question and add it to the list of sampled queries
        question = judge.choose_question()
        sampled_queries.append(question)
        
        # choose all possible nc2 pairs of models
        for model_a, model_b in judge.choose_all_models():
            score_a, score_b = judge.winrate(model_a, question), judge.winrate(model_b, question)
            if score_a is None or score_b is None:
                continue

            # The scoring function here is really simple. We just use
            # the fluency to judge which model produces the best output.
            # However, in reality we would use an LLM judge here.
            if score_a >= score_b:
                preferences.append((model_a, model_b))
            elif score_b > score_a:
                preferences.append((model_b, model_a))
    
    try:
        result = choix.ilsr_pairwise(len(judge.dataset_idx), preferences)
    
    except (ValueError, RuntimeError) as e:
        result = choix.opt_pairwise(len(judge.dataset_idx), preferences)

    return result, sampled_queries

def print_results(tournament_count, mean_scores, median_scores, ci_95, model_names):
    print("=============================================")
    print("G O L D : B E S T   M O D E L S   R A N K E D")
    print("=============================================")
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
    parser.add_argument("--eval_hf_dataset", type=str, required=False, default="nthakur/mirage-eval-rag-output")
    parser.add_argument("--language", type=str, required=False, default="en")
    parser.add_argument("--llm4judge_scores", type=str, required=False, default=None)
    parser.add_argument("--split", type=str, required=False, default="dev")
    parser.add_argument("--cache_dir", type=str, required=False, default="/u3/n3thakur/projects/cache")
    parser.add_argument("--num_tournaments", type=int, default=200, required=False)
    parser.add_argument("--matches_per_tournament", type=int, default=50, required=False)
    parser.add_argument("--regression_model_name", type=str, default=None, required=False)
    parser.add_argument("--output_filepath", type=str, default="avg_citation_map_recall_scores.tsv", required=False)
    args = parser.parse_args()

    FEATURES = [
        "language_detection", 
        "en_detection",
        "citation_MAP@10",
        "citation_Recall@10",
        "Meta-Llama-3-8B-Instruct-fluency",
        "xlni_context_entailment",
        "xlni_context_neutral",
        "bge-reranker-v2-m3-score",
        "Meta-Llama-3-8B-Instruct-answer-eval",
        "answer_bleu",
        "answer_rougeL"
    ]
    lang = args.language
    print(f"Loading feature dataset for Language: {lang} ....")
    load_dataset = datasets.load_dataset(args.eval_hf_dataset, lang, split=args.split, cache_dir=args.cache_dir)

    # Load the predictions and references
    hf_dataset = {}
    
    for row in load_dataset:
        hf_dataset[row["query_id"]] = row

    print(f"Evaluating Language: {lang} ....")
    judge = LLMJudge(MODEL_NAMES, args.llm4judge_scores)
    num_models = len(MODEL_NAMES)
    
    num_tournaments = args.num_tournaments
    matches_per_tournament = args.matches_per_tournament

    all_scores = np.zeros((num_tournaments, num_models))
    train_X = np.zeros((num_tournaments * num_models, len(FEATURES)))
    train_Y = np.zeros((num_tournaments * num_models, 1))

    for tournament_counter in range(num_tournaments):
        params, queries = simulate_tournament(num_matches=matches_per_tournament, judge=judge)
        feature_dict = {model_name: {} for model_name in MODEL_NAMES}
        
        # Take the sum of all the features for each model
        for query_id in queries:
            for feature_name in FEATURES:
                for model_name, feature_value in hf_dataset[query_id][feature_name].items():
                    if feature_value == None: feature_value = 0.0
                    
                    if feature_name not in feature_dict[model_name]:
                        feature_dict[model_name][feature_name] = 0.0
                    
                    if model_name in MODEL_NAMES:
                        feature_dict[model_name][feature_name] += feature_value
        
        # Take the average of all the 50 features
        for model_name in MODEL_NAMES:
            for feature_name in FEATURES:
                if feature_name in feature_dict[model_name]:
                    feature_dict[model_name][feature_name] /= len(queries)
        
        all_scores[tournament_counter] = params
        
        for idx, model_name in enumerate(MODEL_NAMES):
            try:
                train_X[tournament_counter * num_models + idx] = [feature_dict[model_name][feature_name] for feature_name in FEATURES]
                train_Y[tournament_counter * num_models + idx] = params[idx]
            except KeyError:
                print(model_name, feature_dict[model_name])

    mean_scores = np.mean(all_scores, axis=0)
    median_scores = np.median(all_scores, axis=0)
    ci_95 = np.percentile(all_scores, [2.5, 97.5], axis=0)

    # Train Linear regression model
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=seed)
    
    reg = RegressionModel(args.regression_model_name)
    reg.fit(X_train, y_train)


    print(f"Training Score: {reg.score(X_train, y_train)}")
    print(f"Test Score: {reg.score(X_test, y_test)}")
    # print(f"Model Coefficients: {reg.coef_}")
    print(f"Features: {FEATURES}")


    #### Final Gold Ranking ####
    print_results(num_tournaments, mean_scores, median_scores, ci_95, MODEL_NAMES)
    print()

    #### Final ranking prediction using the model ####
    final_scores = {model_name: {} for model_name in MODEL_NAMES}
    
    for model_name in MODEL_NAMES:
        for query_id in hf_dataset:
            all_features = []
            for feature_name in FEATURES:
                feature_value = hf_dataset[query_id][feature_name][model_name]
                if feature_value is not None:
                    all_features.append(feature_value)
                else:
                    all_features.append(0.0)
            
            final_scores[model_name][query_id] = reg.predict(np.array(all_features).reshape(1, -1))[0]
    
    # Rank the models based on the final scores
    final_mean = {model_name: np.mean(list(final_scores[model_name].values())) for model_name in MODEL_NAMES}
    final_ci_95 = {model_name: np.percentile(list(final_scores[model_name].values()), [2.5, 97.5]) for model_name in MODEL_NAMES}
    final_scores = {k: v for k, v in sorted(final_mean.items(), key=lambda item: item[1], reverse=True)}

    
    print("=======================================================")
    print("S I L V E R : F I N A L   L E A R N E D   R A N K I N G")
    print("=======================================================")
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

    # Show distribution of all scores
    # boxplot_data = []
    # for i in range(num_models):
    #     for score in all_scores[:, i]:
    #         boxplot_data.append({'Model': f'{SHORT_MODEL_NAMES[i]}', 'Score': score})

    # # Convert to a DataFrame
    # df = pd.DataFrame(boxplot_data)

    # # Create the box plot
    # plt.figure(figsize=(5, 4))
    # sns.boxplot(x='Model', y='Score', data=df)
    # plt.title('Distribution of Logits for Each Model')
    # plt.xlabel('Model')
    # plt.ylabel('Score')
    # plt.show()

    # # Create a heatmap
    # plt.figure(figsize=(5, 4))
    # sns.heatmap(prob_matrix, annot=True, cmap='viridis', xticklabels=MODEL_NAMES, yticklabels=MODEL_NAMES)
    # plt.title('Probabilities of Model A beating Model B')
    # plt.xlabel('Model B')
    # plt.ylabel('Model A')
    # plt.show()
