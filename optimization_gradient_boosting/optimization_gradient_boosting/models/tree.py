import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
from joblib import Parallel, delayed
import logging
import warnings

from metrics import variance_reduction

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)


@dataclass
class Node:
    """
    Represent a node in the decision tree
    """
    feature: int
    threshold: float
    left_split: 'Node'
    right_split: 'Node'
    gain: float


@dataclass
class Leaf:
    """
    Represent the leaf in a decision tree
    """
    value: float


@dataclass
class DecisionTree:
    """
    Create a decision tree in this case a regression tree
    """
    min_sample_split: int = 2
    max_depth: Optional[int] = None
    max_feature: Optional[int] = None
    min_info_gain: float = 0.0
    n_jobs: int = 0
    tree = None

    @staticmethod
    def _compute_best_feature_threshold(x: np.array, y: np.array, index_feature: int):
        best_split = {}
        best_gain = -np.inf
        thresholds = np.unique(x)
        for threshold in thresholds:
            mask_left = x < threshold
            gain = variance_reduction(y=y, y_left=y[mask_left], y_right=y[~mask_left])
            if gain > best_gain:
                best_split = {
                    'feature': index_feature,
                    'threshold': threshold,
                    'gain': gain
                }
                best_gain = gain
        return best_split

    def get_best_split(self, X: np.array, Y: np.array) -> dict:
        """
        Get the best split given 2 arrays which gives us the best information gain
        """
        nfeatures = X.shape[1]

        feature_group = np.random.choice(
            nfeatures,
            self.max_feature if self.max_feature is not None else nfeatures,
            replace=False
        )
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_best_feature_threshold)(
                x=X[:, index_feature],
                y=Y,
                index_feature=index_feature
            )
            for index_feature in feature_group
        )
        if all(results):
            best_results = max(results, key=lambda x: x.get('gain', -np.inf))
            mask_left = X[:, best_results['feature']] < best_results['threshold']
            return {
                **best_results,
                'left_split_X': X[mask_left, :],
                'left_split_Y': Y[mask_left],
                'right_split_X': X[~mask_left, :],
                'right_split_Y': Y[~mask_left],
            }

    def _grow(self, X: np.array, Y: np.array, depth: int = 0) -> Union[Node, Leaf]:
        """
        Build the tree based on the best splits recursively
        """
        max_depth = self.max_depth if self.max_depth is not None else np.inf
        if depth <= max_depth and len(Y) >= self.min_sample_split:
            best_split = self.get_best_split(X=X, Y=Y)
            if best_split:
                left_child = self._grow(best_split['left_split_X'], best_split['left_split_Y'], depth=depth + 1)
                right_child = self._grow(best_split['right_split_X'], best_split['right_split_Y'], depth=depth + 1)
                return Node(
                    feature=best_split['feature'],
                    threshold=best_split['threshold'],
                    left_split=left_child,
                    right_split=right_child,
                    gain=best_split['gain']
                )
        return Leaf(value=np.mean(Y))

    def fit(self, X: np.array, Y: np.array):
        """ Build the decision tree"""
        self.tree = self._grow(X=X, Y=Y)
        return self

    def _single_predict(self, x: np.array, node: Union[Node, Leaf]) -> np.array:
        """ Predict y given x"""
        if isinstance(node, Leaf):
            return node.value

        feature_index = x[node.feature]

        if feature_index < node.threshold:
            return self._single_predict(x=x, node=node.left_split)
        if feature_index >= node.threshold:
            return self._single_predict(x=x, node=node.right_split)

    def predict(self, X: np.array) -> np.array:
        return np.array([self._single_predict(x=row, node=self.tree) for row in X])
