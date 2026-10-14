"""Run bounded collection shards in fresh Isaac processes and log GPU usage."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g1_embodiedskills.io import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("isaac-python", "sonic", "reference", "asset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=20)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"started_at": datetime.now().astimezone().isoformat(), "shards": [], "completed": False}
    driver = Path(__file__).with_name("run_sim.py").resolve()
    with (args.output / "resources.jsonl").open("w") as resources:
        for index, config in enumerate(args.configs):
            destination = args.output / f"shard-{index:02d}"
            row = {"index": index, "config": str(config.resolve()), "config_sha256": sha256(config),
                   "output": str(destination), "status": "running"}
            report["shards"].append(row)
            write_json(args.output / "pipeline.json", report)
            command = ["bash", str(args.isaac_python.resolve()), str(driver), "--sonic", str(args.sonic.resolve()),
                       "--reference", str(args.reference.resolve()), "--asset", str(args.asset.resolve()),
                       "--output", str(destination), "--mode", "collect", "--seconds", "24", "--cameras",
                       "--num-envs", str(args.num_envs), "--collection-config", str(config.resolve())]
            started, sampled = time.monotonic(), 0.
            print(f"G1_DATASET shard={index} config={config.name} started", flush=True)
            with (args.output / f"shard-{index:02d}.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while process.poll() is None:
                        now = time.monotonic()
                        if now-sampled >= 10:
                            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                                  "--format=csv,noheader,nounits"], capture_output=True, text=True)
                            resources.write(json.dumps({"time": datetime.now().astimezone().isoformat(),
                                "shard": index, "memory_mib_and_util_percent": gpu.stdout.strip()}) + "\n")
                            resources.flush()
                            sampled = now
                        time.sleep(1)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            row.update(returncode=process.returncode, wall_seconds=time.monotonic()-started,
                       status="complete" if process.returncode == 0 else "failed")
            if process.returncode == 0:
                row["result"] = json.loads((destination / "result.json").read_text())
            write_json(args.output / "pipeline.json", report)
            print(f"G1_DATASET shard={index} status={row['status']} result={row.get('result')}", flush=True)
            if process.returncode:
                raise RuntimeError(f"Collection shard {index} failed; see its log")
    report["completed"] = True
    report["finished_at"] = datetime.now().astimezone().isoformat()
    write_json(args.output / "pipeline.json", report)


if __name__ == "__main__":
    main()
