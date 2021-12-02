from typing import Dict, List, Callable

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from pprint import pprint

from models.gboost import GBoost
from models.tree import DecisionTree
from metrics import mse


def main(models: Dict[str, object], X: np.array, y: np.array,
         metrics: Dict[str, Callable[[np.array, np.array], float]]) -> None:
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[model_name] = {
            metric_name: metric(y_test, preds) for metric_name, metric in metrics.items()
        }
    pprint(results)


if __name__ == '__main__':
    X, y = load_diabetes(return_X_y=True)
    main(
        models={
            'custom_tree': DecisionTree(n_jobs=-2),
            'sklearn_tree': DecisionTreeRegressor(),
            'custom_gbm_10': GBoost(n_trees=10, learning_rate=0.01, max_depth=10, n_jobs=-2),
            'sklearn_gbm_10': GradientBoostingRegressor(random_state=19, n_estimators=10, max_depth=10, learning_rate=0.01),
            'custom_gbm_50': GBoost(n_trees=50, learning_rate=0.01, max_depth=5, n_jobs=-2),
            'sklearn_gbm_50': GradientBoostingRegressor(random_state=19, n_estimators=50, max_depth=5, learning_rate=0.01),
            'test_no_change': GBoost(n_trees=50, learning_rate=0.01, max_depth=5, validation_fraction=0.2, n_iter_no_change=4, tol=10.0, n_jobs=-2)
        },
        X=X,
        y=y,
        metrics={'mse': mse, 'mae': mean_absolute_error}
    )
