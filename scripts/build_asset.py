import argparse
import os
from pathlib import Path
from g1_embodiedskills.assets import build_robot

p = argparse.ArgumentParser()
p.add_argument("--sonic", type=Path, default=os.environ.get("SONIC_ROOT"))
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.sonic is None:
    p.error("Set SONIC_ROOT or pass --sonic")
print(build_robot(a.sonic, a.output))
