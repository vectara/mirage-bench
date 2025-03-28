from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class RegressionModel:
    def __init__(self, model_name: str = "RandomForestRegressor"):
        self.model_name = model_name
        if self.model_name == "LinearRegression":
            self.reg_model = LinearRegression()
        elif self.model_name == "RandomForestRegressor":
            self.model = RandomForestRegressor()
        else:
            raise ValueError(
                f"Model {model_name} not supported. Supported models: ['LinearRegression', 'RandomForestRegressor']"
            )

    def fit(self, feature_names: list[str], X: np.ndarray, y: np.ndarray, debug: bool = False):
        self.model.fit(X, y)

        if self.model_name == "RandomForestRegressor":
            feature_importances = self.model.feature_importances_

            # Get the indices of the sorted feature importances
            indices = np.argsort(feature_importances)[::-1]

            if debug:
                # Assuming feature_names is a list of your feature names
                logging.info("===========================================")
                logging.info("R A N D O M   F O R E S T   F E A T U R E S")
                logging.info("===========================================")
                logging.info("   FEATURE                                    IMPORTANCE")
                logging.info("   =====================================================")
                for i in range(len(feature_importances)):
                    logging.info(
                        f"{i + 1:2}. {feature_names[indices[i]]:40}     {feature_importances[indices[i]]:0.5f}"
                    )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.model.score(X, y)
