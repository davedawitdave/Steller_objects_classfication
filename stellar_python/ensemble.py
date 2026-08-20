
from __future__ import annotations

import numpy as np

from stellar_python.decision_tree import DecisionTreeClassifier


class BaggingForest:
    def __init__(
        self, nEstimators=15, maxDepth=10, minSamplesSplit=2, minSamplesLeaf=1, minInformationGain=0.0,
        criterion="gini", maxFeatures="sqrt", seed=0,
    ):
        self.nEstimators = nEstimators
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.minInformationGain = minInformationGain
        self.criterion = criterion
        self.maxFeatures = maxFeatures
        self.seed = seed
        self.trees = []
        self.nClasses = None
        self.outOfBagMask = None



    def resolveMaxFeatures(self, numFeatures):
        if self.maxFeatures is None:
            return None
        if self.maxFeatures == "sqrt":
            return max(1, int(np.sqrt(numFeatures)))
        if self.maxFeatures == "log2":
            return max(1, int(np.log2(numFeatures)))
        return int(self.maxFeatures)



    def fit(self, features, labels, nClasses):
        self.nClasses = nClasses
        numSamples = len(labels)
        maxFeatures = self.resolveMaxFeatures(features.shape[1])
        self.trees = []
        self.outOfBagMask = np.zeros((self.nEstimators, numSamples), dtype=bool)

        for treeIndex in range(self.nEstimators):
            randomGenerator = np.random.default_rng(self.seed + treeIndex)
            bootstrapIndex = randomGenerator.integers(0, numSamples, size=numSamples)
            outOfBagIndex = np.setdiff1d(np.arange(numSamples), bootstrapIndex)
            self.outOfBagMask[treeIndex, outOfBagIndex] = True

            tree = DecisionTreeClassifier(
                maxDepth=self.maxDepth, minSamplesSplit=self.minSamplesSplit,
                minSamplesLeaf=self.minSamplesLeaf, minInformationGain=self.minInformationGain,
                criterion=self.criterion, maxFeatures=maxFeatures, seed=self.seed + treeIndex,
            )
            tree.fit(features[bootstrapIndex], labels[bootstrapIndex], nClasses)
            self.trees.append(tree)
        return self



    def predictProba(self, features):
        allProbabilities = np.stack([tree.predictProba(features) for tree in self.trees], axis=0)
        return allProbabilities.mean(axis=0)



    def predict(self, features):
        return np.argmax(self.predictProba(features), axis=1)



    def outOfBagScore(self, features, labels):
        """Accuracy using, for every sample, only the trees that did not
        see it during their bootstrap draw — a held-out estimate that
        needs no separate validation split.
        """
        numSamples = len(labels)
        voteSum = np.zeros((numSamples, self.nClasses))
        everVoted = np.zeros(numSamples, dtype=bool)
        for treeIndex, tree in enumerate(self.trees):
            outOfBagIndex = np.where(self.outOfBagMask[treeIndex])[0]
            if len(outOfBagIndex) == 0:
                continue
            voteSum[outOfBagIndex] += tree.predictProba(features[outOfBagIndex])
            everVoted[outOfBagIndex] = True
        predictions = np.argmax(voteSum[everVoted], axis=1)
        return float(np.mean(predictions == labels[everVoted]))



    def featureImportances(self):
        return np.mean([tree.featureImportances for tree in self.trees], axis=0)
