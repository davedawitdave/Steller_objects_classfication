# Stellar Classification (SDSS17) — Decision Tree

<table><tr>
<td><img src="https://www.nasa.gov/wp-content/uploads/2023/03/650137main_pia15416b-43_full.jpg?w=1024" width="260"><br><center>Galaxy </center></td>
<td><img src="https://science.nasa.gov/wp-content/uploads/2024/10/quasar-banner-illustration.jpg?w=1024" width="260"><br><center>Quasar </center></td>
<td><img src="https://assets.science.nasa.gov/content/dam/science/missions/hubble/releases/1996/12/STScI-01EVTASFNAT35WSB0DW9XR49ZX.jpg/jcr:content/renditions/800x550.jpg" width="260"><br><center>Star (Betelgeuse)</center></td>
</tr></table>

*Credit: NASA/ESA/Hubble. See `README.md` for sources.*

A from-scratch (no scikit-learn) decision tree and bagging forest that
classify SDSS17 sky survey observations into **GALAXY**, **QSO** (quasar),
or **STAR**, built to be read and explained step by step, following
*Grokking Machine Learning* ch. 9's own order: accuracy, Gini, entropy,
information gain, recursive splitting, then everything built on top of a
working tree.

## Quick start

```
uv sync   # or: pip install numpy pandas matplotlib ipywidgets jupyter
jupyter notebook stellar_classification.ipynb
```

The CSV is already in `data/star_classification.csv`. Run cells top to
bottom — the config path box in section 2 lets you point at a different
copy if needed.

## Why two packages

| | `stellar_metta/` | `stellar_python/` |
|---|---|---|
| contains | the three impurity measures (accuracy/gini/entropy) and their information gain, as **MeTTa** equations | everything else: the tree, the forest, grid search, plotting, I/O — plus a numpy-vectorized copy of the same impurity math |
| written for | one node at a time, read-as-algebra | thousands of candidate splits at once (real tree growth needs the speed) |
| run with | [PeTTa](https://github.com/trueagi-io/PeTTa) | plain Python |

`stellar_python/metta_bridge.py` is the seam between them: it shells out to
PeTTa to evaluate `stellar_metta/spliter.metta` and `info_gain.metta`, so a
notebook cell can check that the hand-written MeTTa formulas and the
numpy-vectorized engine agree on the same numbers — not just assume the
translation was correct.

**PeTTa is not bundled with this project.** Set `metta.petta_dir` in
`config.json` to wherever your local PeTTa checkout lives (a sibling of
`data/`, same as this project's layout). Until it's found there,
`metta_bridge.py` transparently falls back to calling
`impurity_engine.py`/`info_gain.py` directly and prints a one-line notice —
the notebook runs end to end either way, but only exercises the real
`.metta` files with PeTTa in place. This has been verified against a real
PeTTa checkout: its own usage is `sh run.sh <file.metta>`, run from
anywhere, since `run.sh` resolves its own location from `$0`.

```json
"metta": {
  "petta_dir": "PeTTa",
  "run_script": "run.sh",
  "metta_dir": "stellar_metta",
  "files": ["spliter.metta", "info_gain.metta"],
  "command_template": ["sh", "{petta_dir}/{run_script}", "{metta_file}"]
}
```

If your local PeTTa build expects a different invocation than
`sh <petta_dir>/run.sh <metta_file>`, edit `command_template` in
`config.json` — `metta_bridge.py` builds its subprocess call from that list
verbatim. One PeTTa quirk worth knowing if you extend the `.metta` files:
its grounded math has no bare `log` function, only a two-argument
`log-math(Base, X, Out)` predicate, which is why `spliter.metta` defines
`log2` as `(log-math 2 $x)` rather than `(/ (log $x) (log 2))`.

## Package layout

```
stellar_metta/
  spliter.metta        gini/entropy/accuracy impurity, weighted average
  info_gain.metta       information gain for each of the three

stellar_python/
  impurity_engine.py    numpy-vectorized twin of spliter.metta
  info_gain.py           numpy-vectorized gain, used by the split search
  decision_nodes.py     node bookkeeping, vectorized split search,
                         recursive growth, 4 stopping criteria
  decision_tree.py      the public DecisionTreeClassifier
  tree_plot.py           text + matplotlib rendering, metrics in every node
  grid_search.py         hyperparameter search, permutation importance,
                         tree-vs-forest comparison
  regularization.py      LassoRegression (ISTA) — L1 diagnostic only
  ensemble.py             BaggingForest
  preprocessing.py, metrics.py, persistence.py
  metta_bridge.py         calls the .metta files via PeTTa, with a
                         Python fallback

config.json              single source of truth for one run: seed, columns,
                         split fractions, grids, MeTTa/image paths
stellar_classification.ipynb
data/star_classification.csv
models/                  saved pipeline (created by the notebook)
artifacts/tree_plots/    saved tree renderings (created by the notebook)
```

## What the notebook covers

1. **Setup and cleaning** — drop SDSS's `-9999` sentinel rows; drop the
   nine survey/plate/ID columns (`plate`, `MJD`, `fiber_ID`, `run_ID`,
   `rerun_ID`, `cam_col`, `field_ID`, `obj_ID`, `spec_obj_ID`) since they
   record which telescope program found an object, not a physical property
   of it.
2. **Part A — a small worked example.** Whole-set impurity computed two
   ways (numpy and MeTTa) and checked to agree; best single split under
   each of the three criteria; three trees grown and plotted, one per
   criterion; a 4-parameter grid search (`maxDepth`, `minSamplesLeaf`,
   `criterion`, `minInformationGain`); permutation importance; a
   forest-vs-tree check (noisy at this size, on purpose — it's a rehearsal
   for Part B, not a real answer).
3. **Part B — the full ~100k-row dataset.** Correlation pruning and an L1
   (Lasso, via ISTA) regression are both shown as diagnostics on the
   magnitude bands' redundancy — neither is actually applied to the
   feature set, since training on the full data is fast enough not to need
   the reduction. Hyperparameter search on a 10k-row subsample (first
   `maxDepth`/`minSamplesLeaf`/`criterion`, then `minInformationGain` on
   its own, each visualized), then full training, then the tree-vs-forest
   keep/drop decision on real held-out data.
4. **Evaluation** — classification report, confusion matrix, and one-vs-rest
   ROC curves with AUC, for whichever model (tree or forest) the section 3
   comparison kept.
5. **Save/reload** and an **interactive prediction widget** (sliders for
   position, magnitudes, and redshift).

## A design decision worth knowing about

Earlier versions of this project predicted redshift from photometry
(a "Layer 1" regression) and fed that *prediction* to the classifier, to
avoid leaking the true spectroscopic redshift. This version removes that
step: **true redshift is a direct input feature** now. That's a
deliberate simplification, and it matters for interpreting the results —
redshift is close to deterministic for `STAR` (essentially zero) and a
strong signal for `QSO` vs. `GALAXY`, so accuracy is much higher here than
it would be with only position and magnitudes.

## Image credits

All three images are NASA/ESA Hubble Space Telescope releases (public
domain / NASA image use policy): NGC 4258 (M106), quasar 3C 273, and
Betelgeuse.
