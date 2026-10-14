#!/usr/bin/env python3
"""Offline extraction/splitting and upstream EmbodiedSkills training entry point.

Examples (set EMBODIEDSKILLS_ROOT to your upstream checkout first)::

    python scripts/train_selector.py inventory
    python scripts/train_selector.py validate --raw --input artifacts/branches.npz
    python scripts/train_selector.py extract --input artifacts/branches.npz --output artifacts/features.npz
    python scripts/train_selector.py split --input artifacts/features.npz --output-dir artifacts/splits
    python scripts/train_selector.py train --train artifacts/splits/train.npz --validation artifacts/splits/validation.npz --output artifacts/selector

Model loading never downloads weights. Outputs/data/checkpoints stay local.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from g1_embodiedskills.learning import main


if __name__ == "__main__":
    main()
