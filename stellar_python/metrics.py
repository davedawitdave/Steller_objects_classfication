"""Regression and classification metrics, computed without scikit-learn."""
from __future__ import annotations

import numpy as np


def rootMeanSquaredError(trueValues, predictedValues):
    return float(np.sqrt(np.mean((trueValues - predictedValues) ** 2)))



def meanAbsoluteError(trueValues, predictedValues):
    return float(np.mean(np.abs(trueValues - predictedValues)))



def rSquared(trueValues, predictedValues):
    residualSumSquares = np.sum((trueValues - predictedValues) ** 2)
    totalSumSquares = np.sum((trueValues - trueValues.mean()) ** 2)
    return float(1.0 - residualSumSquares / totalSumSquares)



def confusionMatrix(trueLabels, predictedLabels, nClasses):
    """matrix[i, j] = number of samples of true class i predicted as j."""
    matrix = np.zeros((nClasses, nClasses), dtype=int)
    for trueLabel, predictedLabel in zip(trueLabels, predictedLabels):
        matrix[trueLabel, predictedLabel] += 1
    return matrix



def accuracy(trueLabels, predictedLabels):
    return float(np.mean(trueLabels == predictedLabels))



def precisionRecallF1PerClass(matrix):
    """Per-class precision/recall/F1 derived from a confusion matrix.
    precision_c = matrix[c, c] / column sum c (predicted positive)
    recall_c    = matrix[c, c] / row sum c    (actual positive)
    """
    nClasses = matrix.shape[0]
    precision = np.zeros(nClasses)
    recall = np.zeros(nClasses)
    f1 = np.zeros(nClasses)
    support = matrix.sum(axis=1)

    for classIndex in range(nClasses):
        truePositive = matrix[classIndex, classIndex]
        predictedPositive = matrix[:, classIndex].sum()
        actualPositive = matrix[classIndex, :].sum()
        precision[classIndex] = truePositive / predictedPositive if predictedPositive > 0 else 0.0
        recall[classIndex] = truePositive / actualPositive if actualPositive > 0 else 0.0
        denominator = precision[classIndex] + recall[classIndex]
        f1[classIndex] = 2 * precision[classIndex] * recall[classIndex] / denominator if denominator > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "support": support}



def classificationReport(trueLabels, predictedLabels, classNames):
    nClasses = len(classNames)
    matrix = confusionMatrix(trueLabels, predictedLabels, nClasses)
    stats = precisionRecallF1PerClass(matrix)
    total = matrix.sum()
    macroPrecision = stats["precision"].mean()
    macroRecall = stats["recall"].mean()
    macroF1 = stats["f1"].mean()
    weightedF1 = float(np.sum(stats["f1"] * stats["support"]) / total)

    lines = [f"{'class':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"]
    for classIndex, name in enumerate(classNames):
        lines.append(
            f"{name:<10}{stats['precision'][classIndex]:>10.3f}{stats['recall'][classIndex]:>10.3f}"
            f"{stats['f1'][classIndex]:>10.3f}{stats['support'][classIndex]:>10d}"
        )
    lines.append("")
    lines.append(f"{'accuracy':<10}{'':>10}{'':>10}{accuracy(trueLabels, predictedLabels):>10.3f}{total:>10d}")
    lines.append(f"{'macro avg':<10}{macroPrecision:>10.3f}{macroRecall:>10.3f}{macroF1:>10.3f}{total:>10d}")
    lines.append(f"{'weighted avg':<10}{'':>10}{'':>10}{weightedF1:>10.3f}{total:>10d}")
    return "\n".join(lines)



def rocCurve(trueLabels, probabilities, classIndex, nThresholds=100):
    """One-vs-rest ROC curve for classIndex: false-positive rate and
    true-positive rate at nThresholds evenly-spaced cutoffs on the
    predicted probability of classIndex, from 1 (strictest) down to 0
    (everything predicted positive).
    """
    isPositive = (trueLabels == classIndex).astype(int)
    scores = probabilities[:, classIndex]
    thresholds = np.linspace(1.0, 0.0, nThresholds)

    truePositiveRates = np.empty(nThresholds)
    falsePositiveRates = np.empty(nThresholds)
    for i, threshold in enumerate(thresholds):
        predictedPositive = scores >= threshold
        truePositives = np.sum(predictedPositive & (isPositive == 1))
        falsePositives = np.sum(predictedPositive & (isPositive == 0))
        actualPositives = np.sum(isPositive == 1)
        actualNegatives = np.sum(isPositive == 0)
        truePositiveRates[i] = truePositives / actualPositives if actualPositives > 0 else 0.0
        falsePositiveRates[i] = falsePositives / actualNegatives if actualNegatives > 0 else 0.0

    return falsePositiveRates, truePositiveRates, thresholds



def oneVersusRestAuc(trueLabels, probabilities, classIndex):
    """AUC for classIndex vs. the rest, via the Mann-Whitney U statistic:
    AUC = P(score of a random positive > score of a random negative)
        = U / (nPositive * nNegative)
    which matches the trapezoidal-ROC AUC without scanning thresholds.
    """
    isPositive = (trueLabels == classIndex).astype(int)
    scores = probabilities[:, classIndex]
    nPositive = isPositive.sum()
    nNegative = len(isPositive) - nPositive
    if nPositive == 0 or nNegative == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)

    sortedScores = scores[order]
    sortedRanks = ranks[order]
    position = 0
    while position < len(sortedScores):
        tieEnd = position
        while tieEnd + 1 < len(sortedScores) and sortedScores[tieEnd + 1] == sortedScores[position]:
            tieEnd += 1
        if tieEnd > position:
            sortedRanks[position:tieEnd + 1] = sortedRanks[position:tieEnd + 1].mean()
        position = tieEnd + 1
    ranks[order] = sortedRanks

    rankSumPositive = ranks[isPositive == 1].sum()
    uStatistic = rankSumPositive - nPositive * (nPositive + 1) / 2
    return float(uStatistic / (nPositive * nNegative))
