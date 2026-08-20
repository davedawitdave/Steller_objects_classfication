"""Information gain: how much a split reduces impurity relative to its
parent node, for each of the three criteria in impurity_engine.py. A split
is only worth taking if this number clears the configured minInformationGain
threshold (see decision_nodes.py).
"""
from __future__ import annotations

import numpy as np

from stellar_python import impurity_engine


def impurityGain(criterion, parentCounts, leftCounts, rightCounts):
    """parentImpurity - weightedChildImpurity, under the named criterion.

    parentCounts, leftCounts, rightCounts : class-count arrays, shape
    (n_classes,) for one split or (n_candidates, n_classes) for a batch.
    """
    impurityFunction = impurity_engine.IMPURITY_FUNCTIONS[criterion]

    parentImpurity = impurityFunction(parentCounts)
    leftImpurity = impurityFunction(leftCounts)
    rightImpurity = impurityFunction(rightCounts)

    leftCount = np.asarray(leftCounts, dtype=float).sum(axis=-1)
    rightCount = np.asarray(rightCounts, dtype=float).sum(axis=-1)
    childImpurity = impurity_engine.weightedAverage(leftImpurity, rightImpurity, leftCount, rightCount)

    return parentImpurity - childImpurity



def giniGain(parentCounts, leftCounts, rightCounts):
    return impurityGain("gini", parentCounts, leftCounts, rightCounts)



def entropyGain(parentCounts, leftCounts, rightCounts):
    return impurityGain("entropy", parentCounts, leftCounts, rightCounts)



def accuracyGain(parentCounts, leftCounts, rightCounts):
    """Gain in the impurity-shaped accuracy (misclassificationRate), i.e.
    how much the overall accuracy improves once the node is split in two
    and each child predicts its own majority class.
    """
    return impurityGain("accuracy", parentCounts, leftCounts, rightCounts)



def allCriteriaGains(parentCounts, leftCounts, rightCounts):
    """All three gains at once, keyed by criterion name — used to print
    every measure inside a tree-plot node regardless of which criterion
    the tree was actually grown with.
    """
    return {
        criterion: impurityGain(criterion, parentCounts, leftCounts, rightCounts)
        for criterion in impurity_engine.IMPURITY_FUNCTIONS
    }
