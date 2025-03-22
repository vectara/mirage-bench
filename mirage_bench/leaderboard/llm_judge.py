from __future__ import annotations

import logging
import random
from itertools import combinations

import numpy as np
from datasets import Dataset

from .util import load_pairwise_judgments

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAMES = [
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


class LLMJudge:
    def __init__(self, pairwise_dataset: str, model_names: list[str] = None, **kwargs):
        """Initialize the LLMJudge with the model names and pairwise judgments."""
        self.model_names = DEFAULT_MODEL_NAMES if model_names is None else model_names
        # load pairwise judgments
        self.pairwise_results = load_pairwise_judgments(
            model_names=self.model_names, pairwise_dataset=pairwise_dataset, **kwargs
        )
        self.dataset_idx = [i for i in range(len(model_names))]
        self.question_ids = self.get_all_questions()

        logging.info(
            f"Loaded data for {len(self.model_names)} models: {self.model_names}.\nFound {len(self.question_ids)} unique queries."
        )

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
        return None  # For a tie or any other scenario.


def build_training_dataset(
    queries: list[str],
    train_model_names: list[str],
    features: list[str],
    feature_dataset: dict[str, Dataset],
    bt_learned_model: dict[str, float],
    all_model_names: list[str] = DEFAULT_MODEL_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
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
    num_models = len(train_model_names)
    model_to_int = {model: idx for idx, model in enumerate(all_model_names)}

    train_X = np.zeros((num_models, len(features)))
    train_Y = np.zeros((num_models, 1))
    feature_dict = {model_name: {} for model_name in train_model_names}

    # Take the sum of all the features for each model
    for query_id in queries:
        for feature_name in features:
            for model_name, feature_value in feature_dataset[query_id][feature_name].items():
                if model_name in train_model_names:
                    feature_dict[model_name].setdefault(feature_name, 0.0)
                    feature_dict[model_name][feature_name] += feature_value if feature_value else 0.0

    # Take the average of all the features.
    for model_name in train_model_names:
        for feature_name in features:
            if feature_name in feature_dict[model_name]:
                feature_dict[model_name][feature_name] /= len(queries)

    for i, model_name in enumerate(train_model_names):
        idx = model_to_int.get(model_name, None)
        try:
            train_X[i] = [feature_dict[model_name][feature_name] for feature_name in features]
            train_Y[i] = bt_learned_model[idx]

        except KeyError:
            features_not_present = [
                feature_name for feature_name in features if feature_name not in feature_dict[model_name]
            ]
            logger.warning(f"Features not present for model {model_name}: {features_not_present}")

    return train_X, train_Y
