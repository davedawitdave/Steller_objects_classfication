from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from stellar_python import impurity_engine, info_gain

_alreadyWarned = False
_ansiEscape = re.compile(r"\x1b\[[0-9;]*m")


def resolveMettaSettings(config):
  
    mettaConfig = config.get("metta")
    if mettaConfig is None:
        raise KeyError(
            "config.json has no \"metta\" section. Add one, e.g.:\n"
            '  "metta": {"petta_dir": "PeTTa", "run_script": "run.sh", '
            '"metta_dir": "stellar_metta", "files": ["spliter.metta", "info_gain.metta"], '
            '"command_template": ["sh", "{petta_dir}/{run_script}", "{metta_file}"], '
            '"timeout_seconds": 30}'
        )

    if "petta_dir" in mettaConfig and "run_script" in mettaConfig:
        return mettaConfig["petta_dir"], mettaConfig["run_script"]

    if "petta_path" in mettaConfig:
        legacyPath = Path(mettaConfig["petta_path"])
        return str(legacyPath.parent), legacyPath.name

    raise KeyError(
        'config.json\'s "metta" section has neither ("petta_dir" and "run_script") '
        'nor the older "petta_path". Set petta_dir to your local PeTTa checkout '
        '(e.g. "PeTTa") and run_script to "run.sh".'
    )



def mettaFilesText(config):
    mettaDir = Path(config["metta"]["metta_dir"])
    return "\n\n".join((mettaDir / name).read_text() for name in config["metta"]["files"])



def pettaIsAvailable(config):
    pettaDir, runScript = resolveMettaSettings(config)
    return (Path(pettaDir) / runScript).exists()



def runMettaExpression(config, expressionText):
   
    pettaDir, runScript = resolveMettaSettings(config)
    fullProgram = mettaFilesText(config) + f"\n\n!({expressionText})\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".metta", delete=False) as tempFile:
        tempFile.write(fullProgram)
        tempFilePath = tempFile.name

    command = [
        part.format(petta_dir=pettaDir, run_script=runScript, metta_file=tempFilePath)
        for part in config["metta"]["command_template"]
    ]
    timeoutSeconds = config["metta"].get("timeout_seconds", 30)
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeoutSeconds)

    cleanedStdout = _ansiEscape.sub("", result.stdout)
    resultLines = [line for line in cleanedStdout.strip().splitlines() if line.strip()]
    if result.returncode != 0 or not resultLines:
        raise RuntimeError(f"PeTTa produced no usable result.\nstderr:\n{result.stderr}")
    return resultLines[-1].strip()



def warnFallbackOnce():
    global _alreadyWarned
    if not _alreadyWarned:
        print("(PeTTa not found at the configured metta.petta_dir — using the equivalent "
              "Python fallback in impurity_engine.py/info_gain.py; set metta.petta_dir in "
              "config.json to your local PeTTa checkout to run the real .metta files.)")
        _alreadyWarned = True



def giniImpurity(c0, c1, c2, config):
    if pettaIsAvailable(config):
        return float(runMettaExpression(config, f"gini-impurity {c0} {c1} {c2}"))
    warnFallbackOnce()
    return float(impurity_engine.giniImpurity([c0, c1, c2]))



def entropyImpurity(c0, c1, c2, config):
    if pettaIsAvailable(config):
        return float(runMettaExpression(config, f"entropy-impurity {c0} {c1} {c2}"))
    warnFallbackOnce()
    return float(impurity_engine.entropyImpurity([c0, c1, c2]))



def accuracyScore(c0, c1, c2, config):
    if pettaIsAvailable(config):
        return float(runMettaExpression(config, f"accuracy-score {c0} {c1} {c2}"))
    warnFallbackOnce()
    return float(impurity_engine.accuracyScore([c0, c1, c2]))



def giniGain(parentCounts, leftCounts, rightCounts, config):
    args = list(parentCounts) + list(leftCounts) + list(rightCounts)
    if pettaIsAvailable(config):
        return float(runMettaExpression(config, "gini-gain " + " ".join(str(a) for a in args)))
    warnFallbackOnce()
    return float(info_gain.giniGain(parentCounts, leftCounts, rightCounts))



def entropyGain(parentCounts, leftCounts, rightCounts, config):
    args = list(parentCounts) + list(leftCounts) + list(rightCounts)
    if pettaIsAvailable(config):
        return float(runMettaExpression(config, "entropy-gain " + " ".join(str(a) for a in args)))
    warnFallbackOnce()
    return float(info_gain.entropyGain(parentCounts, leftCounts, rightCounts))