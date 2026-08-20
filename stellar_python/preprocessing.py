"""Data preparation utilities. Every function that involves randomness
takes an explicit seed. No scikit-learn is used anywhere here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# SDSS pipeline flag: a photometric magnitude of -9999 marks a band the
# pipeline could not measure. It is not a real magnitude and must not enter
# any distance- or split-based computation.
SENTINEL_MAGNITUDE = -9999.0


def dropBadPhotometry(dataframe, magnitudeColumns):
    """Remove rows where any photometric band equals the SDSS sentinel value."""
    mask = np.ones(len(dataframe), dtype=bool)
    for column in magnitudeColumns:
        mask &= dataframe[column].to_numpy() > -1000.0
    return dataframe.loc[mask].reset_index(drop=True)



def trainValTestSplit(sampleCount, valFraction, testFraction, seed):
    """Index arrays (train, val, test) via one seeded permutation, so the
    same seed reproduces the exact same three partitions every run.
    """
    randomGenerator = np.random.default_rng(seed)
    shuffledIndex = randomGenerator.permutation(sampleCount)
    valCount = int(round(sampleCount * valFraction))
    testCount = int(round(sampleCount * testFraction))
    valIndex = shuffledIndex[:valCount]
    testIndex = shuffledIndex[valCount:valCount + testCount]
    trainIndex = shuffledIndex[valCount + testCount:]
    return trainIndex, valIndex, testIndex



def stratifiedSubsample(labels, sampleCount, seed):
    """Index array of size sampleCount preserving the class proportions of
    labels, used both for the 20-row book-style demo and the hyperparameter
    search subsample.
    """
    randomGenerator = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    take = np.round(proportions * sampleCount).astype(int)
    take[np.argmax(take)] += sampleCount - take.sum()

    chosenIndices = []
    for classValue, classTake in zip(classes, take):
        classIndex = np.where(labels == classValue)[0]
        classTake = min(classTake, len(classIndex))
        chosenIndices.append(randomGenerator.choice(classIndex, size=classTake, replace=False))
    combined = np.concatenate(chosenIndices)
    randomGenerator.shuffle(combined)
    return combined



class StandardScaler:
    """Feature standardization: (x - mean) / std, fit on training data
    only, so no validation/test statistics leak into the transform.
    """

    def __init__(self):
        self.mean = None
        self.std = None



    def fit(self, features):
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self



    def transform(self, features):
        return (features - self.mean) / self.std



    def fitTransform(self, features):
        return self.fit(features).transform(features)



def correlationMatrix(dataframe, columns):
    return dataframe[columns].corr().to_numpy()



def decorrelateFeatures(dataframe, columns, threshold):
    """Greedy correlation pruning: scan every pair of columns in order,
    and whenever |correlation| exceeds threshold, drop the second column
    of the pair (if not already dropped). This is the "regression targets
    are highly correlated -> collapse them" step, done by elimination
    rather than by fitting a combiner model, since dropping a column is
    the more transparent of the two for magnitudes that are this
    collinear (see the correlation heatmap in the notebook).
    """
    matrix = correlationMatrix(dataframe, columns)
    dropped = []

    for i in range(len(columns)):
        if columns[i] in dropped:
            continue
        for j in range(i + 1, len(columns)):
            if columns[j] in dropped:
                continue
            if abs(matrix[i, j]) > threshold:
                dropped.append(columns[j])

    keep = [column for column in columns if column not in dropped]
    return keep, dropped



def oneHotEncode(dataframe, column):
    """Generic one-hot helper, provided for completeness: none of the
    pipeline's current features are categorical (position and magnitudes
    are all continuous), so this is unused in the notebook today but is
    here in case a future categorical feature needs it.
    """
    return pd.get_dummies(dataframe[column], prefix=column)
