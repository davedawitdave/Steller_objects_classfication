"""Save/reload the fitted pipeline (scaler + regressor + tree/forest) plus
the config and metrics it was trained with, so a run is reproducible from
the saved artifact and readable without unpickling anything.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path


def savePipeline(path, pipeline, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "pipeline.pkl", "wb") as fileHandle:
        pickle.dump(pipeline, fileHandle)
    with open(path / "metadata.json", "w") as fileHandle:
        json.dump(metadata, fileHandle, indent=2, default=str)



def loadPipeline(path):
    path = Path(path)
    with open(path / "pipeline.pkl", "rb") as fileHandle:
        pipeline = pickle.load(fileHandle)
    with open(path / "metadata.json", "r") as fileHandle:
        metadata = json.load(fileHandle)
    return pipeline, metadata
