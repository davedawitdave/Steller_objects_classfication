"""The performance-critical half of the scratch (no scikit-learn) machine
learning stack: numpy-vectorized tree building, grid search, plotting,
regularization diagnostics, and I/O. The single-node impurity math these
modules are built on is defined twice on purpose — once here as a
vectorized numpy engine (impurity_engine.py, for speed on real data) and
once in ../stellar_metta/ as plain MeTTa equations (for readability, one
node at a time). metta_bridge.py is what lets a notebook cell call the
MeTTa version directly, via PeTTa, to check the two agree.
"""
