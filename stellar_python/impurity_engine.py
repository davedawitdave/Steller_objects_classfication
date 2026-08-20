"""Numpy-vectorized twin of ../stellar_metta/spliter.metta: the same three
formulas (accuracy, Gini, entropy), but every function accepts either one
node's counts (a 1D array of length n_classes) or a batch of candidate
nodes (a 2D array of shape (n_candidates, n_classes)). Real tree growth has
to score thousands of candidate thresholds per node, which is what this
vectorized version is for; the MeTTa file is the one-node-at-a-time,
read-as-algebra version used for teaching and for the notebook's
Python-vs-MeTTa cross-check in Part A.
"""
from __future__ import annotations

import numpy as np


def classProportions(counts):
    """Turn class counts into class proportions: p_c = counts_c / total.

    counts : array of shape (..., n_classes). The proportions are taken
    along the last axis, so this works for one node or a batch of nodes.
    """
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=-1, keepdims=True)
    return counts / totals



def giniImpurity(counts):
    """Gini index: 1 - sum(p_c^2). Zero when the node is pure (one class),
    maximal when classes are evenly mixed.
    """
    proportions = classProportions(counts)
    return 1.0 - np.sum(proportions ** 2, axis=-1)



def entropyImpurity(counts):
    """Shannon entropy: -sum(p_c * log2(p_c)), with the 0 * log2(0) = 0
    convention so an absent class does not produce a NaN.
    """
    proportions = classProportions(counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(proportions > 0, proportions * np.log2(proportions), 0.0)
    return np.clip(-np.sum(terms, axis=-1), 0.0, None)  # clip away floating-point -0.0



def accuracyScore(counts):
    """Accuracy of the single best strategy available at a node: always
    predict its majority class. accuracy = max_c(counts_c) / sum(counts).
    """
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=-1)
    majority = counts.max(axis=-1)
    return majority / totals



def misclassificationRate(counts):
    """1 - accuracyScore: an impurity-shaped version of accuracy (higher
    means worse), so it combines with weightedAverage the same way Gini
    and entropy do, and a positive "gain" always means "improved" for all
    three criteria.
    """
    return 1.0 - accuracyScore(counts)



def weightedAverage(leftValue, rightValue, leftCount, rightCount):
    """Combine a left/right child statistic weighted by how many samples
    landed in each child: (leftCount * leftValue + rightCount * rightValue)
    / (leftCount + rightCount). Used to turn two children's impurity into
    the single number a split is judged by.
    """
    leftCount = np.asarray(leftCount, dtype=float)
    rightCount = np.asarray(rightCount, dtype=float)
    totalCount = leftCount + rightCount
    return (leftCount * leftValue + rightCount * rightValue) / totalCount



# Dispatch table: criterion name -> the node-level function that scores it.
# "accuracy" is stored as its impurity form (misclassificationRate) so every
# criterion in this table is "lower is purer", matching Gini and entropy.
IMPURITY_FUNCTIONS = {
    "gini": giniImpurity,
    "entropy": entropyImpurity,
    "accuracy": misclassificationRate,
}
