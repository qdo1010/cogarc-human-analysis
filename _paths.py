"""Common path resolution for the human-analysis modules.

In this standalone repo the project root IS the repo root — this file
sits at the top of the repo. Set the environment variable
``ARC_GNN_ROOT`` to an ARC-GNN checkout if you want ``human_solver.py``
and ``run_human_analysis.py`` (which import ``enhanced_egnn`` / ``solver``
from that parent project) to work.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "CogARC-dataRepository"
ARTIFACT_ROOT = PROJECT_ROOT / "prior_analysis"
TRAINING_DIR = PROJECT_ROOT / "training"


# Put the repo root on sys.path so bare ``import human_targets`` works
# under both direct-script and ``python -m`` invocation.
_ROOT_STR = str(PROJECT_ROOT)
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

# Optional: if ARC_GNN_ROOT points at a valid directory, expose it so
# the two GNN-integration scripts can find the ARC-GNN core modules.
_ARC_GNN = os.environ.get("ARC_GNN_ROOT")
if _ARC_GNN and Path(_ARC_GNN).is_dir() and _ARC_GNN not in sys.path:
    sys.path.insert(0, _ARC_GNN)


def ensure_cwd() -> None:
    """Pin the working directory to the repo root so relative output paths
    (``prior_analysis/...``) resolve regardless of invocation location."""
    if Path.cwd().resolve() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)


def artifact(*parts: str) -> str:
    return str(ARTIFACT_ROOT.joinpath(*parts))
