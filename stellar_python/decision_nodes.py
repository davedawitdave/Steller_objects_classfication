"""Recursive tree growth: finding the best split at a node, deciding when
to stop, and building the TreeNode structure that decision_tree.py and
tree_plot.py both read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stellar_python import info_gain, impurity_engine


@dataclass
class TreeNode:

    sampleCount: int
    classCounts: np.ndarray
    depth: int
    giniValue: float
    entropyValue: float
    accuracyValue: float
    majorityClass: int
    isLeaf: bool = True
    stopReason: str | None = None
    featureIndex: int | None = None
    threshold: float | None = None
    splitGain: float | None = None
    splitCriterion: str | None = None
    leftChild: "TreeNode | None" = None
    rightChild: "TreeNode | None" = None



def describeNode(labels, nClasses, depth):
    """Build the diagnostics for one node from its labels, without
    deciding yet whether it will end up a leaf or a split.
    """
    counts = np.zeros(nClasses)
    values, freqs = np.unique(labels, return_counts=True)
    counts[values] = freqs
    return TreeNode(
        sampleCount=len(labels),
        classCounts=counts,
        depth=depth,
        giniValue=float(impurity_engine.giniImpurity(counts)),
        entropyValue=float(impurity_engine.entropyImpurity(counts)),
        accuracyValue=float(impurity_engine.accuracyScore(counts)),
        majorityClass=int(np.argmax(counts)),
    )



def bestSplitForFeature(featureColumn, labels, nClasses, criterion, minSamplesLeaf, minInformationGain):
    """Best threshold on a single feature, scored under one criterion.

    Returns (threshold, gain) or None if no valid split exists: 
    """
    sampleCount = len(labels)
    order = np.argsort(featureColumn, kind="mergesort")
    sortedFeature = featureColumn[order]
    sortedOneHot = np.eye(nClasses)[labels[order]]

    leftCounts = np.cumsum(sortedOneHot, axis=0)[:-1]
    totalCounts = leftCounts[-1] + sortedOneHot[-1]
    rightCounts = totalCounts - leftCounts

    leftSize = np.arange(1, sampleCount)
    rightSize = sampleCount - leftSize

    validSplit = (
        (sortedFeature[:-1] != sortedFeature[1:])
        & (leftSize >= minSamplesLeaf)
        & (rightSize >= minSamplesLeaf)
    )
    if not np.any(validSplit):
        return None

    gains = info_gain.impurityGain(criterion, totalCounts, leftCounts, rightCounts)
    gains = np.where(validSplit, gains, -np.inf)

    bestPosition = int(np.argmax(gains))
    bestGain = float(gains[bestPosition])
    if bestGain <= minInformationGain:
        return None

    threshold = float((sortedFeature[bestPosition] + sortedFeature[bestPosition + 1]) / 2.0)
    return threshold, bestGain



def bestSplitOverall(features, labels, nClasses, criterion, minSamplesLeaf, minInformationGain, featureSubset):

    best = None
    bestGain = minInformationGain
    for featureIndex in featureSubset:
        result = bestSplitForFeature(
            features[:, featureIndex], labels, nClasses, criterion, minSamplesLeaf, minInformationGain
        )
        if result is None:
            continue
        threshold, gain = result
        if gain > bestGain:
            bestGain = gain
            best = (featureIndex, threshold, gain)
    return best



def shouldStopSplitting(labels, depth, maxDepth, minSamplesSplit):
  
    if np.all(labels == labels[0]):
        return "pure node"
    if len(labels) < minSamplesSplit:
        return "below minSamplesSplit"
    if maxDepth is not None and depth >= maxDepth:
        return "reached maxDepth"
    return None



def growTree(
    features, labels, nClasses, criterion, maxDepth, minSamplesSplit, minSamplesLeaf,
    minInformationGain=0.0, depth=0, maxFeatures=None, randomGenerator=None,
):
    """Recursively build a TreeNode. maxFeatures/randomGenerator are only
    used by the forest (ensemble.py): each split then considers a random
    subset of columns instead of all of them.
    """
    node = describeNode(labels, nClasses, depth)

    earlyStopReason = shouldStopSplitting(labels, depth, maxDepth, minSamplesSplit)
    if earlyStopReason is not None:
        node.stopReason = earlyStopReason
        return node

    numFeatures = features.shape[1]
    if maxFeatures is None or maxFeatures >= numFeatures:
        featureSubset = np.arange(numFeatures)
    else:
        featureSubset = randomGenerator.choice(numFeatures, size=maxFeatures, replace=False)

    split = bestSplitOverall(
        features, labels, nClasses, criterion, minSamplesLeaf, minInformationGain, featureSubset
    )
    if split is None:
        node.stopReason = "no split cleared minInformationGain"
        return node

    featureIndex, threshold, gain = split
    goLeft = features[:, featureIndex] <= threshold

    node.isLeaf = False
    node.featureIndex = featureIndex
    node.threshold = threshold
    node.splitGain = gain
    node.splitCriterion = criterion
    node.leftChild = growTree(
        features[goLeft], labels[goLeft], nClasses, criterion, maxDepth,
        minSamplesSplit, minSamplesLeaf, minInformationGain, depth + 1, maxFeatures, randomGenerator,
    )
    node.rightChild = growTree(
        features[~goLeft], labels[~goLeft], nClasses, criterion, maxDepth,
        minSamplesSplit, minSamplesLeaf, minInformationGain, depth + 1, maxFeatures, randomGenerator,
    )
    return node



def treeDepth(node):
    if node.isLeaf:
        return 0
    return 1 + max(treeDepth(node.leftChild), treeDepth(node.rightChild))



def leafCount(node):
    if node.isLeaf:
        return 1
    return leafCount(node.leftChild) + leafCount(node.rightChild)



def featureImportanceFromNode(node, numFeatures, importances=None):
    """Mean-decrease-impurity feature importance: at every split, credit
    its feature with sampleCount * splitGain, then normalize to sum to 1.
    """
    if importances is None:
        importances = np.zeros(numFeatures)
    if not node.isLeaf:
        importances[node.featureIndex] += node.sampleCount * node.splitGain
        featureImportanceFromNode(node.leftChild, numFeatures, importances)
        featureImportanceFromNode(node.rightChild, numFeatures, importances)
    return importances
