"""
Usage:

python3 ./22Jul2024/weight_learning_bootstrap_elo_judge.py \
    --eval_hf_dataset nthakur/mirage-eval-rag-output \
    --llm4judge_scores ./data/pairwise_gpt4o_llm_as_a_judge_ja_scores.jsonl \
    --split dev \
    --language ja \
    --regression_model_name RandomForestRegressor \
    --output_filepath ./data/avg_citation_map_recall_scores.tsv \
    --matches_per_tournament 100

"""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from itertools import combinations
import json
import random

import choix
import scipy.stats as stats
import tqdm
import numpy as np

from typing import List
import datasets, argparse

seed = 42

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


HOLDOUT_MODELS = {'google/gemma-1.1-2b-it', 'meta-llama/Meta-Llama-3-70B-Instruct'}

# Create mappings
model_to_int = {model: idx for idx, model in enumerate(MODEL_NAMES)}
int_to_model = {idx: model for idx, model in enumerate(MODEL_NAMES)}

# Function to get integer value for a given model name
def get_int_from_model(model_name):
    return model_to_int.get(model_name, None)  # Return None if model_name is not found

# Function to get model name for a given integer value
def get_model_from_int(int_value):
    return int_to_model.get(int_value, None)  # Return None if int_value is not found


class RegressionModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if self.model_name == "LinearRegression":
            self.model = LinearRegression()
        elif self.model_name == "RandomForestRegressor":
            self.model = RandomForestRegressor()

    def fit(self, feature_names, X, y, debug=False):
        self.model.fit(X, y)
        if self.model_name == "RandomForestRegressor":
            feature_importances = self.model.feature_importances_

            # Get the indices of the sorted feature importances
            indices = np.argsort(feature_importances)[::-1]

            if debug:
                # Assuming feature_names is a list of your feature names
                print("===========================================")
                print("R A N D O M   F O R E S T   F E A T U R E S")
                print("===========================================")
                print()
                print("   FEATURE                                    IMPORTANCE")
                print("   =====================================================")
                for i in range(len(feature_importances)):
                    print(f"{i + 1:2}. {feature_names[indices[i]]:40}     {feature_importances[indices[i]]:0.5f}")

    
    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
    

def parse_comparisons(pairwise_comparisons) -> List[tuple]:
    """Accept the pairwise_outputs dictionary for a question.
    Returns:
      - a list of tuples: (model_a, model_b, winner)

    Winner can be 0, 1, or -1 (tie).
    """
    results = []
    for key, result in pairwise_comparisons.items():
        model_a, model_b = key.split(",")
        if result is None:
            continue

        result_tail = result[-25:]
        model_a_wins = "[[A]]" in result_tail
        model_b_wins = "[[B]]" in result_tail
        tie = "[[C]]" in result_tail

        if not model_a_wins and not model_b_wins and not tie:
            continue
        if model_a_wins:
            results.append((
                get_int_from_model(model_a),
                get_int_from_model(model_b),
                0))
        if model_b_wins:
            results.append((
                get_int_from_model(model_a),
                get_int_from_model(model_b),
                1))
        if tie:
            results.append((
                get_int_from_model(model_a),
                get_int_from_model(model_b),
                -1))
    return results


class LLMJudge:
    def __init__(self, model_names: List[str], llm4judge_result: List[List[str]]):
        def load_json_file(filepath):
            results = {}
            with open(filepath, 'r') as file:
                for line in file:
                    question = json.loads(line)

                    results[question["query_id"]] = {}
                    for (model_a, model_b, result) in parse_comparisons(question["pairwise_outputs"]):
                        results[question["query_id"]][(model_a, model_b)] = result
            return results

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
        """Return all models."""
        return list(combinations(self.dataset_idx, 2))

    def choose_question(self):
        """Randomly choose a question id."""
        return random.choice(list(self.question_ids))


    def all_preference_pairs(self, question_id):
        comparisons = self.pairwise_results.get(question_id, {})
        for (model_a_idx, model_b_idx), result in comparisons.items():
            if result == 0:
                yield (model_a_idx, model_b_idx)
            elif result == 1:
                yield (model_b_idx, model_a_idx)


    def preference_pair(self, question_id, model_a_idx, model_b_idx):
        """Given a question, return the winning model.

        Returns:
          - (model_a_idx, model_b_idx) if model A was better.
          - (model_b_idx, model_a_idx) if model B was better.
          - None if the models were tied, or the question id is invalid.
        """

        comparisons = self.pairwise_results.get(question_id, {})
        key = (model_a_idx, model_b_idx)
        if key not in comparisons:
            key = (model_b_idx, model_a_idx)
            if key not in comparisons:
                return None

        result = comparisons[key]
        if result == 0:
            return (model_a_idx, model_b_idx)
        elif result == 1:
            return (model_b_idx, model_a_idx)
        return None    # For a tie or any other scenario.


def simulate_tournament(
    num_matches=-1, judge=None):
    """Simulate a tournament with the specified number of head-to-head matchups.

    Args:
      num_matches: The number of matches. If set to -1, use all questions exactly
          once. This uses all available information to train the model.

    """
    preferences, sampled_queries = [], []
    
    if num_matches > 0:
        for idx in range(num_matches):
            # Sample a question, with replacement, and add it to the list
            # of sampled queries.
            question = judge.choose_question()
            sampled_queries.append(question)
            
            # Choose all possible nc2 pairs of models.
            preferences.extend([i for i in judge.all_preference_pairs(question)])
    elif num_matches == -1:
        for question in judge.question_ids:
            sampled_queries.append(question)

            # Choose all possible nc2 pairs of models.
            preferences.extend([i for i in judge.all_preference_pairs(question)])

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
    print("    MODEL NAME                                MEAN          95% CI")
    print("    ==============================================================")
    for i, model_idx in enumerate(reversed(np.argsort(mean_scores))):
        minus = ci_95[0][model_idx] - mean_scores[model_idx]
        plus = ci_95[1][model_idx] - mean_scores[model_idx]
        print(f"{i + 1:2}. {model_names[model_idx]:<40} {mean_scores[model_idx]: 4.3f}     {minus: 3.2f}/+{plus:3.2f}")

def p_win(pair, params):
    # Placeholder for the actual choix.probabilities function
    # Assume it returns a tuple (prob_i, prob_j) for the models in 'pair'
    i, j = pair
    return choix.probabilities([i, j], params)


def rank_sample_analysis(llm_judge_results: List[str]):
    """Analyse the relationship between ranking accuracy and sample performance."""


    def tournament(comparisons) -> List[int]:
        """Returns the indexes of models, from best to worst."""
        preferences = []
        for (model_a, model_b, winner) in comparisons:
            if winner == 0:
                preferences.append((model_a, model_b))
            elif winner == 1:
                preferences.append((model_b, model_a))
            elif winner == -1:
                # preferences.append((model_a, model_b))
                # preferences.append((model_b, model_a))
                pass
        try:
            result = choix.ilsr_pairwise(len(MODEL_NAMES), preferences)
        except (ValueError, RuntimeError) as e:
            result = choix.opt_pairwise(len(MODEL_NAMES), preferences)

        return reversed(np.argsort(result))


    with open(llm_judge_results, 'r') as file:
        pairwise_comparisons = []
        for line in tqdm.tqdm(file):
            question = json.loads(line)
            pairwise_comparisons.extend(parse_comparisons(question["pairwise_outputs"]))

        full_results = [i for i in tournament(pairwise_comparisons)]

        num_trials = 100
        print(f"Based on a population of {len(pairwise_comparisons)} pairwise comparisons.")
        print(f"Confidence intervals based on {num_trials} trials of the sampling procedure.")
        print()
        print("   SAMPLE RATE      KENDALL'S τ COEFFICIENT           95% CI")
        print("   =========================================================")


        for sample_rate in [i / 100 for i in range(5, 101, 5)]:
            all_scores = []
            for _ in range(num_trials):
                partial_results = tournament(
                    random.sample(pairwise_comparisons, int(sample_rate * len(pairwise_comparisons))))
                partial_results = [i for i in partial_results]
                kendall_tau, kendall_p_value = stats.kendalltau(
                    full_results, partial_results)
                all_scores.append(kendall_tau)

            mean_score = np.mean(all_scores, axis=0)
            ci_95 = np.percentile(all_scores, [2.5, 97.5], axis=0)

            minus = ci_95[0] - mean_score
            plus = ci_95[1] - mean_score
            print(f"         {sample_rate:0.3f}                      {mean_score:-0.5f}      {minus: 3.2f}/+{plus:3.2f}")
    print("done")


def build_training_dataset(queries, models, features, bt_learned_model):
    """

    Params:
        - queries, a list of query ids
        - models, a list of model names
        - features, a list of features to use for training
        - bt_learned_model, a Bradley-Terry model. The y-labels are the logits.
            This needs to be logits for all the models.

    Returns:
        - train_X, the features
        - train_Y, the labels
    """
    num_models = len(models)
    assert num_models <= len(MODEL_NAMES)
    assert len(bt_learned_model) == len(MODEL_NAMES)

    train_X = np.zeros((num_models, len(features)))
    train_Y = np.zeros((num_models, 1))

    feature_dict = {model_name: {} for model_name in models}
        
    # Take the sum of all the features for each model
    for query_id in queries:
        for feature_name in features:
            for model_name, feature_value in hf_dataset[query_id][feature_name].items():
                if model_name in models:
                    feature_dict[model_name].setdefault(feature_name, 0.0)
                    feature_dict[model_name][feature_name] += (feature_value if feature_value else 0.0)
    
    # Take the average of all the features.
    for model_name in models:
        for feature_name in features:
            if feature_name in feature_dict[model_name]:
                feature_dict[model_name][feature_name] /= len(queries)
    
    for i, model_name in enumerate(models):
        idx = get_int_from_model(model_name)
        try:
            train_X[i] = [feature_dict[model_name][feature_name] for feature_name in features]
            train_Y[i] = bt_learned_model[idx]
        except KeyError:
            print(model_name, feature_dict[model_name])
    return train_X, train_Y


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_hf_dataset", type=str, required=False, default="nthakur/mirage-eval-rag-output")
    parser.add_argument("--language", type=str, required=False, default="en")
    parser.add_argument("--llm4judge_scores", type=str, required=False, default=None)
    parser.add_argument("--split", type=str, required=False, default="dev")
    parser.add_argument("--cache_dir", type=str, required=False, default="/home/amin/clients/zir/amin/nthakur/cache")
    parser.add_argument("--num_tournaments", type=int, default=200, required=False)
    parser.add_argument("--matches_per_tournament", type=int, default=50, required=False)
    parser.add_argument("--regression_model_name", type=str, default=None, required=False)
    parser.add_argument("--output_filepath", type=str, default="../data/avg_citation_map_recall_scores.tsv", required=False)
    args = parser.parse_args()

    lang = args.language
    
    # print(f"Performing Rank Sampling Analysis: ...")
    # rank_sample_analysis(args.llm4judge_scores)
    # sys.exit(0)

    print(f"Evaluating Language: {lang} ....")
    judge = LLMJudge(MODEL_NAMES, args.llm4judge_scores)

    # This dataset is https://huggingface.co/datasets/nthakur/mirage-eval-rag-output.

    print(f"Loading feature dataset for Language: {lang} ....")
    hf_dataset = {
        row["query_id"]: row
        for row in datasets.load_dataset(
            args.eval_hf_dataset,
            lang,
            split=args.split,
            cache_dir=args.cache_dir
        )
    }

    print(f"Evaluating Language: {lang} ....")
    judge = LLMJudge(MODEL_NAMES, args.llm4judge_scores)
    num_models = len(MODEL_NAMES)
    
    num_tournaments = args.num_tournaments
    matches_per_tournament = args.matches_per_tournament

    all_scores = np.zeros((num_tournaments, num_models))
    r2_errors = []
    for tournament_counter in range(num_tournaments):
        learned_model, queries = simulate_tournament(
            num_matches=matches_per_tournament, judge=judge)
        all_scores[tournament_counter] = learned_model

        train_X, train_Y = build_training_dataset(
            queries=queries,
            models=set(MODEL_NAMES) - HOLDOUT_MODELS,
            features=FEATURES,
            bt_learned_model=learned_model)

        predict_X, predict_Y = build_training_dataset(
            queries=queries,
            models=HOLDOUT_MODELS,
            features=FEATURES,
            bt_learned_model=learned_model)
        
        reg = RegressionModel(args.regression_model_name)
        reg.fit(FEATURES, train_X, train_Y.ravel())
        r2_errors.append(reg.score(predict_X, predict_Y))

    mean_scores = np.mean(all_scores, axis=0)
    median_scores = np.median(all_scores, axis=0)
    ci_95 = np.percentile(all_scores, [2.5, 97.5], axis=0)


    #### Final Gold Ranking ####
    print_results(num_tournaments, mean_scores, median_scores, ci_95, MODEL_NAMES)
    print()

    # Build the final prediction model using all available data.

    #### Final ranking prediction using the model ####

    train_X, train_Y = build_training_dataset(
        queries=[query_id for query_id in hf_dataset],
        models=set(MODEL_NAMES) - HOLDOUT_MODELS,
        features=FEATURES,
        bt_learned_model=mean_scores)

    predict_X, predict_Y = build_training_dataset(
        queries=[query_id for query_id in hf_dataset],
        models=HOLDOUT_MODELS,
        features=FEATURES,
        bt_learned_model=mean_scores)

    rank_X, _ = build_training_dataset(
        queries=[query_id for query_id in hf_dataset],
        models=MODEL_NAMES,
        features=FEATURES,
        bt_learned_model=mean_scores)
        
    reg = RegressionModel(args.regression_model_name)
    reg.fit(FEATURES, train_X, train_Y.ravel(), debug=True)
    print()
    print(f"train R^2={reg.score(train_X, train_Y):0.5f}  holdout R^2={reg.score(predict_X, predict_Y):0.5f}")

    r2_ci_95 = np.percentile(r2_errors, [2.5, 97.5], axis=0)
    r2_mean = np.mean(r2_errors)
    print(f"R^2 Error Analysis ({num_tournaments} models): mean={r2_mean} 95% CI={r2_ci_95[0]-r2_mean:0.4f}/+{r2_ci_95[1]-r2_mean:0.4f}")
    
    rank_Y = reg.predict(rank_X)
    ranked_Y = np.argsort(-rank_Y)

    print()
    print("=========================================")
    print("F I N A L   L E A R N E D   R A N K I N G")
    print("=========================================")
    print()
    print("   MODEL NAME                                 LOGIT      HOLDOUT?")
    print("   ==============================================================")
    for i, idx in enumerate(ranked_Y):
        print(f"{i + 1:2}. {get_model_from_int(idx):<40} {rank_Y[idx]: 3.3f}       {'YES' if get_model_from_int(idx) in HOLDOUT_MODELS else ''}")
    print("\n\n")
