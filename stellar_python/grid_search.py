"""Everything built on top of the tree/forest classes: exhaustive
hyperparameter search, permutation importance for feature selection, and
the rule that decides whether the ensemble is worth keeping at all.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from stellar_python.decision_tree import DecisionTreeClassifier
from stellar_python.ensemble import BaggingForest
from stellar_python import metrics


def searchTreeHyperparameters(
    trainFeatures, trainLabels, valFeatures, valLabels, nClasses,
    maxDepths, minSamplesLeaves, criteria, minInformationGains, seed,
):
    """Exhaustive grid search over (maxDepth, minSamplesLeaf, criterion,
    minInformationGain) — all four stopping criteria at once. Prints one
    line per combination as it runs, so the search's progress is visible
    rather than a silent wait for the final table.
    """
    combinations = [
        (depth, leaf, criterion, minGain)
        for depth in maxDepths for leaf in minSamplesLeaves
        for criterion in criteria for minGain in minInformationGains
    ]
    rows = []

    for combinationIndex, (depth, leaf, criterion, minGain) in enumerate(combinations, start=1):
        startTime = time.time()
        tree = DecisionTreeClassifier(
            maxDepth=depth, minSamplesLeaf=leaf, criterion=criterion,
            minInformationGain=minGain, seed=seed,
        )
        tree.fit(trainFeatures, trainLabels, nClasses)
        fitSeconds = time.time() - startTime

        trainAccuracy = metrics.accuracy(trainLabels, tree.predict(trainFeatures))
        valAccuracy = metrics.accuracy(valLabels, tree.predict(valFeatures))

        print(f"  [{combinationIndex:>3}/{len(combinations)}] depth={str(depth):>4} leaf={leaf:>3} "
              f"criterion={criterion:<9} minGain={minGain:<6} val_acc={valAccuracy:.4f} ({fitSeconds:.2f}s)")

        rows.append(dict(
            maxDepth=depth, minSamplesLeaf=leaf, criterion=criterion, minInformationGain=minGain,
            trainAccuracy=trainAccuracy, valAccuracy=valAccuracy,
            fitSeconds=fitSeconds, leafCount=tree.leafCount(),
        ))

    return pd.DataFrame(rows).sort_values("valAccuracy", ascending=False).reset_index(drop=True)



def searchForestHyperparameters(
    trainFeatures, trainLabels, valFeatures, valLabels, nClasses,
    nEstimatorsList, maxFeaturesList, depth, leaf, minGain, criterion, seed,
):
    combinations = [(n, mf) for n in nEstimatorsList for mf in maxFeaturesList]
    rows = []

    for combinationIndex, (nEstimators, maxFeatures) in enumerate(combinations, start=1):
        startTime = time.time()
        forest = BaggingForest(
            nEstimators=nEstimators, maxDepth=depth, minSamplesLeaf=leaf, minInformationGain=minGain,
            criterion=criterion, maxFeatures=maxFeatures, seed=seed,
        )
        forest.fit(trainFeatures, trainLabels, nClasses)
        fitSeconds = time.time() - startTime

        valAccuracy = metrics.accuracy(valLabels, forest.predict(valFeatures))
        outOfBagAccuracy = forest.outOfBagScore(trainFeatures, trainLabels)

        print(f"  [{combinationIndex:>3}/{len(combinations)}] n_estimators={nEstimators:>3} "
              f"max_features={str(maxFeatures):<6} val_acc={valAccuracy:.4f} oob_acc={outOfBagAccuracy:.4f} ({fitSeconds:.2f}s)")

        rows.append(dict(
            nEstimators=nEstimators, maxFeatures=maxFeatures,
            valAccuracy=valAccuracy, outOfBagAccuracy=outOfBagAccuracy, fitSeconds=fitSeconds,
        ))

    return pd.DataFrame(rows).sort_values("valAccuracy", ascending=False).reset_index(drop=True)



def permutationImportance(model, features, labels, featureNames, seed, nRepeats=5):
    """How much validation accuracy drops when one feature column is
    shuffled (destroying its relationship with the label) while every
    other column stays intact. Repeated nRepeats times per feature and
    averaged, since a single shuffle is noisy.
    """
    randomGenerator = np.random.default_rng(seed)
    baselineAccuracy = metrics.accuracy(labels, model.predict(features))

    importances = []
    for columnIndex, name in enumerate(featureNames):
        drops = []
        for _ in range(nRepeats):
            shuffledFeatures = features.copy()
            shuffledFeatures[:, columnIndex] = randomGenerator.permutation(shuffledFeatures[:, columnIndex])
            shuffledAccuracy = metrics.accuracy(labels, model.predict(shuffledFeatures))
            drops.append(baselineAccuracy - shuffledAccuracy)
        importances.append(dict(feature=name, meanAccuracyDrop=np.mean(drops), stdAccuracyDrop=np.std(drops)))

    return pd.DataFrame(importances).sort_values("meanAccuracyDrop", ascending=False).reset_index(drop=True)



def selectTopFeatures(importanceTable, topK):
    return importanceTable.head(topK)["feature"].tolist()



def compareTreeVsForest(treeValAccuracy, forestValAccuracy, minimumImprovement=0.0):
    """The rule the notebook uses to decide whether ensemble.py earns its
    place in the final pipeline: the forest must beat the single tree on
    validation data by at least minimumImprovement.
    """
    improvement = forestValAccuracy - treeValAccuracy
    keepForest = improvement > minimumImprovement
    return keepForest, improvement
