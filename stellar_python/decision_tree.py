"""Public decision tree class. Growth and node bookkeeping live in
decision_nodes.py; this module is just the fit/predict surface the
notebook and grid_search.py call.
"""
from __future__ import annotations

import numpy as np

from stellar_python import decision_nodes


class DecisionTreeClassifier:
    """A binary decision tree over continuous features, grown by whichever
    of accuracy / gini / entropy is passed as criterion.
    """

    def __init__(
        self, maxDepth=None, minSamplesSplit=2, minSamplesLeaf=1, minInformationGain=0.0,
        criterion="gini", maxFeatures=None, seed=0,
    ):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.minInformationGain = minInformationGain
        self.criterion = criterion
        self.maxFeatures = maxFeatures
        self.seed = seed
        self.root = None
        self.nClasses = None
        self.nFeatures = None
        self.featureImportances = None



    def fit(self, features, labels, nClasses):
        self.nClasses = nClasses
        self.nFeatures = features.shape[1]
        randomGenerator = np.random.default_rng(self.seed)

        self.root = decision_nodes.growTree(
            features, labels, nClasses, self.criterion, self.maxDepth,
            self.minSamplesSplit, self.minSamplesLeaf, self.minInformationGain,
            depth=0, maxFeatures=self.maxFeatures, randomGenerator=randomGenerator,
        )

        rawImportances = decision_nodes.featureImportanceFromNode(self.root, self.nFeatures)
        total = rawImportances.sum()
        self.featureImportances = rawImportances / total if total > 0 else rawImportances
        return self



    def predictRowProbability(self, row):
        node = self.root
        while not node.isLeaf:
            node = node.leftChild if row[node.featureIndex] <= node.threshold else node.rightChild
        return node.classCounts / node.sampleCount



    def predictProba(self, features):
        return np.array([self.predictRowProbability(row) for row in features])



    def predict(self, features):
        return np.argmax(self.predictProba(features), axis=1)



    def depth(self):
        return decision_nodes.treeDepth(self.root)



    def leafCount(self):
        return decision_nodes.leafCount(self.root)
