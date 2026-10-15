#!/usr/bin/env python3
"""Train the preregistered four M1 models using immutable Stage 4 features."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g1_embodiedskills.m1_learning import main

if __name__ == "__main__":
    main()
