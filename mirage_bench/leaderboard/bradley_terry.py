from __future__ import annotations

import choix

from .llm_judge import LLMJudge


def simulate_tournament(num_matches: int = -1, judge: LLMJudge = None):
    """Simulate a tournament with the specified number of head-to-head matchups.

    Args:
      num_matches: The number of matches. If set to -1, use all questions exactly
      once. This uses all available information to train the model.

    """
    preferences, sampled_queries = [], []

    if num_matches > 0:
        for idx in range(num_matches):
            # Bootstrap the model with a random sampled questions.
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
    except (ValueError, RuntimeError):
        result = choix.opt_pairwise(len(judge.dataset_idx), preferences)

    return result, sampled_queries
