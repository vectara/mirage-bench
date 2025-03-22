"""
export CUDA_VISIBLE_DEVICES=-1
export HF_HOME=/mnt/users/n3thakur/cache
export DATASETS_HF_HOME=/mnt/users/n3thakur/cache
export MODEL_NAMES=(
    "Qwen/Qwen2-1.5B-Instruct"
    "Qwen/Qwen2-7B-Instruct"
    "cohereforai-c4ai-aya-23"
    "cohereforai-c4ai-command-r"
    "cohereforai-c4ai-command-r-plus"
)
export FEATURES=(
    'target_language'
    'english_language'
    'citation_MAP@10'
    'citation_Recall@10'
    'answer_bleu'
    'answer_rougeL'
    'support_entailment'
    'support_neutral'
    'reranker_score'
    'answer_overlap_score'
    'fluency_score'
)

for lang in en; do
    python training_and_inference.py --language $lang --split dev \
    --prediction_dataset "nthakur/mirage-bench-output" \
    --pairwise_judgment_dataset "nthakur/mirage-bench-pairwise-judgments" \
    --all_models "${MODEL_NAMES[@]}" \
    --features "${FEATURES[@]}" \
    --holdout_models "cohereforai-c4ai-command-r-plus" \
    --num_tournaments 200 \
    --matches_per_tournament 100 \
    --surrogate_judge RandomForestRegressor
done
"""

import argparse
import logging

import datasets
import numpy as np
from tqdm.auto import tqdm

from mirage_bench import LoggingHandler
from mirage_bench.leaderboard import LLMJudge, RegressionModel, build_training_dataset, simulate_tournament

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout


def get_gold_rankings(mean_scores: np.ndarray, model_names: list[str]) -> dict:
    gold_ranks = {}
    for i, model_idx in enumerate(reversed(np.argsort(mean_scores))):
        gold_ranks[model_names[model_idx]] = i + 1
    return gold_ranks


def beautify_results(tournament_count: int, mean_scores: np.ndarray, ci_95: np.ndarray, model_names: list[str]):
    logging.info("=============================================")
    logging.info("G O L D : B E S T   M O D E L S   R A N K E D")
    logging.info("=============================================")
    logging.info(f"Num Tournaments: {tournament_count}")
    logging.info("    MODEL NAME                                MEAN          95% CI")
    logging.info("    ==============================================================")
    for i, model_idx in enumerate(reversed(np.argsort(mean_scores))):
        minus = ci_95[0][model_idx] - mean_scores[model_idx]
        plus = ci_95[1][model_idx] - mean_scores[model_idx]
        logging.info(
            f"{i + 1:2}. {model_names[model_idx]:<40} {mean_scores[model_idx]: 4.3f}     {minus: 3.2f}/+{plus:3.2f}"
        )


def beautify_predicted_results(
    rank_Y: np.ndarray,
    ranked_Y: np.ndarray,
    gold_rankings: dict,
    get_model_from_int: callable,
    holdout_models: list[str],
):
    logging.info("=========================================")
    logging.info("F I N A L   L E A R N E D   R A N K I N G")
    logging.info("=========================================")
    logging.info("   MODEL NAME                                 LOGIT      GOLD      HOLDOUT?")
    logging.info("   ========================================================================")
    for i, idx in enumerate(ranked_Y):
        gold_rank = gold_rankings[int_to_model.get(idx, None)]
        logging.info(
            f"{i + 1:2}. {int_to_model.get(idx, None):<40} {rank_Y[idx]: 3.3f}      {gold_rank:2}          {'YES' if int_to_model.get(idx, None) in holdout_models else ''}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--prediction_dataset", default=None)
    parser.add_argument("--pairwise_judgment_dataset", default=None)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    parser.add_argument("--all_models", type=str, nargs="+", required=True)
    parser.add_argument("--features", type=str, nargs="+", required=True)
    parser.add_argument("--holdout_models", type=str, nargs="+", default=[])
    parser.add_argument("--num_tournaments", type=int, default=200, required=False)
    parser.add_argument("--matches_per_tournament", type=int, default=50, required=False)
    parser.add_argument("--surrogate_judge", type=str, default=None, required=False)
    args = parser.parse_args()

    # Load the evaluator
    logging.info("=========================================")
    logging.info(f"L A N G U A G E : {args.language}")
    logging.info("=========================================")

    # Load the HF dataset containing the feature values
    feature_dataset = {
        row["query_id"]: row
        for row in datasets.load_dataset(
            args.prediction_dataset, args.language, split=args.split, cache_dir=args.cache_dir
        )
    }

    logging.info(f"Number of tournaments: {args.num_tournaments}")
    logging.info(f"Matches per tournament: {args.matches_per_tournament}")

    r2_errors = []
    num_models = len(args.all_models)
    int_to_model = {idx: model for idx, model in enumerate(args.all_models)}

    llm_judge = LLMJudge(
        pairwise_dataset=args.pairwise_judgment_dataset,
        model_names=args.all_models,
        split=args.split,
        language=args.language,
        cache_dir=args.cache_dir,
    )
    all_scores = np.zeros((args.num_tournaments, num_models))

    for tournament_counter in tqdm(range(args.num_tournaments), desc="Tournaments", total=args.num_tournaments):
        # Simulate a tournament and learn the Bradley-Terry model
        bradley_terry_model, queries = simulate_tournament(num_matches=args.matches_per_tournament, judge=llm_judge)
        all_scores[tournament_counter] = bradley_terry_model

        # Build the training dataset for the regression model
        train_X, train_Y = build_training_dataset(
            queries=queries,
            train_model_names=list(set(args.all_models) - set(args.holdout_models)),
            features=args.features,
            feature_dataset=feature_dataset,
            bt_learned_model=bradley_terry_model,
            all_model_names=args.all_models,
        )
        # logging.info(f"Training dataset: X.shape: {train_X.shape}, Y.shape: {train_Y.shape}")
        reg = RegressionModel(args.surrogate_judge)
        reg.fit(args.features, train_X, train_Y.ravel())

        # Build the holdout dataset for the regression model
        if args.holdout_models:
            predict_X, predict_Y = build_training_dataset(
                queries=queries,
                train_model_names=args.holdout_models,
                features=args.features,
                feature_dataset=feature_dataset,
                bt_learned_model=bradley_terry_model,
                all_model_names=args.all_models,
            )
            # logging.info(f"Holdout dataset: X.shape: {predict_X.shape}, Y.shape: {predict_Y.shape}")
            r2_errors.append(reg.score(predict_X, predict_Y))

    mean_scores = np.mean(all_scores, axis=0)
    ci_95 = np.percentile(all_scores, [2.5, 97.5], axis=0)

    r2_ci_95 = np.percentile(r2_errors, [2.5, 97.5], axis=0)
    r2_mean = np.mean(r2_errors)
    logging.info(
        f"R^2 Error Analysis ({args.num_tournaments} models): mean={r2_mean} 95% CI={r2_ci_95[0] - r2_mean:0.4f}/+{r2_ci_95[1] - r2_mean:0.4f}"
    )

    #### Final Gold Ranking ####
    beautify_results(args.num_tournaments, mean_scores, ci_95, args.all_models)

    # # Build the final prediction model using all available data.

    #### Final ranking prediction using the model ####
    train_X, train_Y = build_training_dataset(
        queries=[query_id for query_id in feature_dataset],
        train_model_names=list(set(args.all_models) - set(args.holdout_models)),
        features=args.features,
        feature_dataset=feature_dataset,
        bt_learned_model=mean_scores,
        all_model_names=args.all_models,
    )

    predict_X, predict_Y = build_training_dataset(
        queries=[query_id for query_id in feature_dataset],
        train_model_names=args.holdout_models,
        features=args.features,
        feature_dataset=feature_dataset,
        bt_learned_model=mean_scores,
        all_model_names=args.all_models,
    )

    rank_X, _ = build_training_dataset(
        queries=[query_id for query_id in feature_dataset],
        train_model_names=args.all_models,
        features=args.features,
        feature_dataset=feature_dataset,
        bt_learned_model=mean_scores,
        all_model_names=args.all_models,
    )

    reg = RegressionModel(args.surrogate_judge)
    reg.fit(args.features, train_X, train_Y.ravel(), debug=True)
    logging.info(f"train R^2={reg.score(train_X, train_Y):0.5f}  holdout R^2={reg.score(predict_X, predict_Y):0.5f}")

    rank_Y = reg.predict(rank_X)
    ranked_Y = np.argsort(-rank_Y)

    gold_rankings = get_gold_rankings(mean_scores, args.all_models)
    beautify_predicted_results(rank_Y, ranked_Y, gold_rankings, int_to_model, args.holdout_models)
