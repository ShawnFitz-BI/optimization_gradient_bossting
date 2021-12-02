import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
import logging
import warnings

from models.tree import DecisionTree

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)


@dataclass
class GBoost:
    """Gradient boosting machine"""
    learning_rate: float
    n_trees: int
    max_depth: Optional[int] = None
    min_sample_split: int = 2
    max_feature: Optional[int] = None
    min_info_gain: float = 0.0
    validation_fraction: float = 0.0
    n_iter_no_change: Optional[int] = np.inf
    tol: float = 1e-4
    seed: int = 19
    n_jobs: int = 0

    trees: Optional[List[DecisionTree]] = field(init=False,
                                                default_factory=list)  # creee nouvelle list a chaque instance
    F0: Optional[float] = field(init=False, default=None)

    def fit(self, X: np.array, Y: np.array):
        """Build the gradient boosting machine"""
        if self.validation_fraction > 0.0:
            mask = np.random.RandomState(seed=self.seed).rand(len(X)) <= self.validation_fraction
            train_X, train_Y = X[mask, :], Y[mask]
            valid_X, valid_Y = X[~mask, :], Y[~mask]
            iter_no_change = 0
        else:
            train_X, train_Y = X, Y
            iter_no_change = -np.inf
        self.F0 = np.mean(train_Y)
        residual = self.F0
        for i in range(self.n_trees):  # si tu utilise pas i mettre un underscore
            if iter_no_change > self.n_iter_no_change:
                logging.info(f'Fitting stopped due to number of iterations without no change')
                break
            logging.info(f'fitting tree number {i}')
            tree = DecisionTree(max_depth=self.max_depth, min_sample_split=self.min_sample_split,
                                max_feature=self.max_feature, min_info_gain=self.min_info_gain, n_jobs=self.n_jobs)
            tree.fit(X=train_X, Y=train_Y - residual)
            increment_residual = self.learning_rate * tree.predict(X=train_X)
            residual += increment_residual

            if self.validation_fraction > 0.0 and np.mean(self.predict(X=valid_X) - valid_Y) < self.tol:
                logging.info(f'iteration without change: {iter_no_change}/{self.n_iter_no_change}')
                iter_no_change += 1
            self.trees.append(tree)
        return self

    def predict(self, X: np.array) -> np.array:
        """Predict y given x"""
        return self.F0 + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)
