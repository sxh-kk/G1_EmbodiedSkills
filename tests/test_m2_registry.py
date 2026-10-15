"""Verify the complete scientific roster and the true companion GPU budget."""
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import sys

import pytest

project = Path(os.environ.get("G1_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(project/"scripts"))
sys.path.insert(0, str(project/"src"))
# The parent needs the newly staged independent statistics module while the
# M1 workspace is frozen. After merge, this points at the ordinary package.
import g1_embodiedskills
staged = Path(__file__).parents[1]/"src/g1_embodiedskills"
if staged.is_dir() and str(staged) not in g1_embodiedskills.__path__:
    g1_embodiedskills.__path__.append(str(staged))
source = Path(__file__).parents[1]/"scripts/run_m2.py"
spec = spec_from_file_location("run_m2", source)
m = module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize("envs,total", [(8, 4320), (10, 3600)])
def test_all_200_anchors_and_noise_cost_are_retained(envs, total):
    protocol = {"num_envs": envs, "amplitude_m": .01, "body_pitch_rad": .03,
                "max_amplitude_halvings": 3, "prefix_tolerance": {"default": 1e-6}}
    shards = m.registry(protocol, "formal", m.CONDITIONS, m.ANCHOR_STEPS)
    assert len(shards) == 20
    planned, physical, selected = [], 0, []
    for shard in shards:
        for batch in shard["config"]["batches"]:
            assert len(batch["anchors"])+len(batch["companion_seeds"]) == envs
            noise = [anchor for anchor in batch["anchors"] if anchor["noise_repeat"]]
            selected += [(anchor["seed"], anchor["anchor_step"]) for anchor in noise]
            planned += [(anchor["seed"], anchor["anchor_step"]) for anchor in batch["anchors"]]
            physical += envs*9*(3 if noise else 1)
    assert len(planned) == len(set(planned)) == 200
    assert len(selected) == 10 and len({seed for seed, _ in planned}) == 40
    assert physical == total
    assert 9*len(planned)+18*len(selected) == 1980


def test_formal_cannot_silently_omit_a_hard_condition():
    protocol = {"num_envs": 8}
    with pytest.raises(ValueError, match="all four conditions"):
        m.registry(protocol, "formal", ["low"], m.ANCHOR_STEPS)
