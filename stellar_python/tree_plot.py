"""Render a TreeNode structure two ways: as indented text (quick to read
in a notebook cell's output) and as a matplotlib figure with every node's
sample count, majority class, and all three impurity measures printed
inside its box, plus the split condition and gain on internal nodes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from stellar_python import decision_nodes


def formatNodeLabel(node, featureNames, classNames):
    className = classNames[node.majorityClass]
    lines = [
        f"n={node.sampleCount}  class={className}",
        f"gini={node.giniValue:.3f}  entropy={node.entropyValue:.3f}  acc={node.accuracyValue:.3f}",
    ]
    if not node.isLeaf:
        featureName = featureNames[node.featureIndex]
        lines.append(f"{featureName} <= {node.threshold:.3f}")
        lines.append(f"gain({node.splitCriterion})={node.splitGain:.4f}")
    else:
        lines.append(f"stop: {node.stopReason}")
    return "\n".join(lines)



def treeToText(node, featureNames, classNames, indent=""):
    label = formatNodeLabel(node, featureNames, classNames).replace("\n", "  |  ")
    lines = [indent + label]
    if not node.isLeaf:
        lines.append(indent + "L->")
        lines.extend(treeToText(node.leftChild, featureNames, classNames, indent + "    ").splitlines())
        lines.append(indent + "R->")
        lines.extend(treeToText(node.rightChild, featureNames, classNames, indent + "    ").splitlines())
    return "\n".join(lines)



def assignLeafPositions(node, nextLeafSlot):
    """Post-order walk: leaves get consecutive integer x-slots, an
    internal node's x is the midpoint of its children's x. Returns the
    node's x position; nextLeafSlot is a one-element list used as a
    mutable counter across the recursion.
    """
    if node.isLeaf:
        x = nextLeafSlot[0]
        nextLeafSlot[0] += 1.4  # extra horizontal gap so neighboring boxes never touch
        node.plotX = x
        return x

    leftX = assignLeafPositions(node.leftChild, nextLeafSlot)
    rightX = assignLeafPositions(node.rightChild, nextLeafSlot)
    node.plotX = (leftX + rightX) / 2.0
    return node.plotX



def drawNode(axes, node, featureNames, classNames):
    y = -node.depth
    label = formatNodeLabel(node, featureNames, classNames)
    faceColor = "#DDEBF7" if node.isLeaf else "#FCE4D6"

    axes.text(
        node.plotX, y, label, ha="center", va="center", fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=faceColor, edgecolor="black", linewidth=0.7),
    )

    if not node.isLeaf:
        for child in (node.leftChild, node.rightChild):
            axes.plot([node.plotX, child.plotX], [y - 0.08, -child.depth + 0.08], color="gray", linewidth=0.8, zorder=0)
        drawNode(axes, node.leftChild, featureNames, classNames)
        drawNode(axes, node.rightChild, featureNames, classNames)



def plotTree(node, featureNames, classNames, title, savePath=None):
    assignLeafPositions(node, nextLeafSlot=[0])
    depth = decision_nodes.treeDepth(node)
    leaves = decision_nodes.leafCount(node)

    figure, axes = plt.subplots(figsize=(max(6, leaves * 2.1), max(4, (depth + 1) * 1.5)))
    drawNode(axes, node, featureNames, classNames)

    axes.set_xlim(-1, leaves * 1.4)
    axes.set_ylim(-depth - 1, 1)
    axes.axis("off")
    axes.set_title(title, fontsize=11)
    plt.tight_layout()

    if savePath is not None:
        figure.savefig(savePath, dpi=150)
        print(f"saved {savePath}")
    plt.show()
    return figure
