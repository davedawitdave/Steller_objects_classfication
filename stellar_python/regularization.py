"""L1 (Lasso) regularization, kept as a standalone diagnostic tool.
"""
from __future__ import annotations

import numpy as np


def designMatrix(features):
    """Prepend an intercept column of ones."""
    ones = np.ones((features.shape[0], 1))
    return np.hstack([ones, features])



class LassoRegression:
    """L1-regularized linear regression via ISTA (iterative soft-thresholding).
    """

    def __init__(self, l1Lambda=0.01, stepSize=0.1, maxIterations=500, tolerance=1e-6, logEvery=50):
        self.l1Lambda = l1Lambda
        self.stepSize = stepSize
        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.logEvery = logEvery
        self.weights = None
        self.lossHistory = []



    def computeLoss(self, designed, targets, weights):
        residual = designed @ weights - targets
        squaredError = 0.5 * np.mean(residual ** 2)
        l1Penalty = self.l1Lambda * np.sum(np.abs(weights[1:]))  # exclude intercept
        return squaredError + l1Penalty



    def softThreshold(self, value, thresholdAmount):
        return np.sign(value) * np.maximum(np.abs(value) - thresholdAmount, 0.0)



    def fit(self, features, targets, verbose=True):
        designed = designMatrix(features)
        numSamples, numColumns = designed.shape
        weights = np.zeros(numColumns)
        self.lossHistory = []

        for iteration in range(self.maxIterations):
            residual = designed @ weights - targets
            gradient = designed.T @ residual / numSamples
            weights = weights - self.stepSize * gradient

            shrunkTail = self.softThreshold(weights[1:], self.stepSize * self.l1Lambda)
            weights = np.concatenate([[weights[0]], shrunkTail])  # intercept untouched

            currentLoss = self.computeLoss(designed, targets, weights)
            self.lossHistory.append(currentLoss)

            if verbose and (iteration % self.logEvery == 0 or iteration == self.maxIterations - 1):
                nonZero = int(np.sum(np.abs(weights[1:]) > 0))
                print(f"  iteration {iteration:4d}  loss={currentLoss:.6f}  nonzero_weights={nonZero}")

            if iteration > 0 and abs(self.lossHistory[-2] - currentLoss) < self.tolerance:
                if verbose:
                    print(f"  converged at iteration {iteration} (loss change below tolerance)")
                break

        self.weights = weights
        return self



    def predict(self, features):
        return designMatrix(features) @ self.weights



    def selectedFeatureMask(self):
        """Boolean mask over the original (non-intercept) columns: True
        where the L1 penalty left a nonzero coefficient.
        """
        return np.abs(self.weights[1:]) > 0
