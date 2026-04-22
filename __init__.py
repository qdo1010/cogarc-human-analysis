"""Human-behavior analysis of the CogARC Experiment 1/2 dataset.

Submodules are importable as ``cogarc_human_analysis.<name>`` (e.g.
``from cogarc_human_analysis.human_targets import human_targets``).

Side effect on import: the parent project root is prepended to
``sys.path`` so scripts in this folder can still import the ARC-GNN core
modules (``enhanced_egnn``, ``solver``, etc.) without installation.
"""

from ._paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, TRAINING_DIR  # noqa
