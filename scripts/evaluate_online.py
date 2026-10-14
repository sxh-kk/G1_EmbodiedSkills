#!/usr/bin/env python3
"""Run the fixed three-method, 30-seed online engineering acceptance protocol.

Each method starts a fresh Isaac process. A fresh parent output directory is
mandatory; there is no resume, retry, seed substitution or log rewriting.
Recorded task failures remain in the denominator. Process failures, incomplete
records, changed inputs and paired-audit failures stop the pipeline.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from g1_embodiedskills.io import sha256, write_json

METHODS = ("direct", "random", "selector")
GPU_QUERY = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"]


class PipelineInterrupted(RuntimeError):
    pass


def timestamp():
    return datetime.now().astimezone().isoformat()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_config(config):
    expected = {"repeats": 30, "num_envs": 8, "seconds": 24, "position_jitter_m": .02,
                "amplitude_m": .025, "execution_steps": 10}
    for name, value in expected.items():
        actual = config.get(name)
        if isinstance(actual, bool) or actual != value:
            raise ValueError(f"Fixed evaluation protocol requires {name}={value}; got {actual!r}")
    if config.get("methods") != list(METHODS):
        raise ValueError("evaluation.json must specify methods in order: direct, random, selector")
    seed = config.get("seed_base")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 900000:
        raise ValueError("evaluation.json needs an integer held-out seed_base >= 900000")
    return config


def source_files(args, config_path):
    """Hash actual executable Python sources without importing simulator modules."""
    files = {"configuration/evaluation.json": config_path}
    for label, root, package in (("project", PROJECT, PROJECT / "src/g1_embodiedskills"),
                                 ("embodiedskills", args.embodiedskills_root,
                                  args.embodiedskills_root / "src/embodiedskills"),
                                 ("sonic", args.sonic, args.sonic / "gear_sonic")):
        if not package.is_dir():
            raise FileNotFoundError(f"Required source package is missing: {package}")
        for path in sorted(package.rglob("*.py")):
            files[f"{label}/{path.relative_to(root)}"] = path
    for path in sorted((PROJECT / "scripts").glob("*.py")):
        files[f"project/{path.relative_to(PROJECT)}"] = path
    return files


def input_files(args):
    files = {"isaac_python_wrapper": args.isaac_python, "selector_checkpoint": args.selector,
             "sonic_checkpoint": args.sonic / "sonic_release/last.pt",
             "sonic_checkpoint_config": args.sonic / "sonic_release/config.yaml", "robot_asset": args.asset}
    for name in ("motion.pkl", "reference.json", "planned.npz", "task.json"):
        files[f"reference/{name}"] = args.reference / name
    return files


def fingerprint(files):
    records = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {name}: {path}")
        records[name] = {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
    encoded = json.dumps({key: value["sha256"] for key, value in records.items()}, sort_keys=True).encode()
    return {"files": records, "combined_sha256": hashlib.sha256(encoded).hexdigest()}


def changed_inputs(original, files):
    changes = []
    for name, path in files.items():
        expected = original["files"].get(name)
        if not path.is_file():
            changes.append({"name": name, "path": str(path), "error": "missing"})
        else:
            actual = sha256(path)
            if expected is None or actual != expected["sha256"]:
                changes.append({"name": name, "path": str(path), "before": None if expected is None else expected["sha256"],
                                "after": actual})
    for name in original["files"].keys()-files.keys():
        changes.append({"name": name, "error": "source file removed from inventory"})
    return changes


def command_for(args, config, method, destination):
    command = ["bash", str(args.isaac_python), str(PROJECT / "scripts/run_sim.py"),
        "--sonic", str(args.sonic), "--reference", str(args.reference), "--asset", str(args.asset),
        "--embodiedskills-root", str(args.embodiedskills_root), "--output", str(destination),
        "--mode", "closed_loop", "--method", method, "--seed-base", str(config["seed_base"]),
        "--repeats", str(config["repeats"]), "--num-envs", str(config["num_envs"]),
        "--seconds", str(config["seconds"]), "--position-jitter", str(config["position_jitter_m"]),
        "--amplitude-m", str(config["amplitude_m"]), "--cameras", "--record-video"]
    if method == "selector":
        command.extend(("--selector", str(args.selector)))
    return command


def sample_gpu(stream, row, started):
    """Preserve the actual query output, return code and any telemetry failure."""
    record = {"time": timestamp(), "stage": row["name"], "pid": row.get("pid"),
              "stage_elapsed_seconds": time.monotonic()-started, "command": GPU_QUERY,
              "scope": "Entire GPU, including RTX and other processes; sampled, not a continuous process-specific peak"}
    try:
        query = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=5, check=False)
        record.update(returncode=query.returncode, stdout=query.stdout, stderr=query.stderr)
        if query.returncode:
            record["error"] = "nvidia-smi returned a nonzero exit code"
        else:
            gpus = []
            for fields in csv.reader(io.StringIO(query.stdout)):
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(f"Unexpected nvidia-smi CSV row: {fields!r}")
                index, name, total, used, utilization = (value.strip() for value in fields)
                gpus.append({"index": int(index), "name": name, "memory_total_mib": float(total),
                             "memory_used_mib": float(used), "utilization_gpu_percent": float(utilization)})
            if not gpus:
                raise ValueError("nvidia-smi returned no GPU rows")
            record["gpus"] = gpus
            peaks = row.setdefault("sampled_gpu_maxima", {})
            for gpu in gpus:
                previous = peaks.setdefault(str(gpu["index"]), {"name": gpu["name"],
                    "memory_total_mib": gpu["memory_total_mib"], "memory_used_mib": 0., "utilization_gpu_percent": 0.})
                for name in ("memory_used_mib", "utilization_gpu_percent"):
                    previous[name] = max(previous[name], gpu[name])
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record.setdefault("returncode", None)
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
    stream.flush()
    row["resource_samples"] = row.get("resource_samples", 0)+1
    if "error" in record:
        row["resource_error"] = record["error"]
        raise RuntimeError(f"GPU resource measurement failed; actual query record preserved: {record['error']}")


def stop_process(process, row):
    if process.poll() is not None:
        return
    row["termination_requested_at"] = timestamp()
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        row["kill_requested_at"] = timestamp()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_process(command, log_path, row, pipeline, output, resources):
    row.update(status="running", command=command, log=str(log_path), started_at=timestamp())
    write_json(output / "pipeline.json", pipeline)
    print(f"G1_EVALUATION stage={row['name']} started", flush=True)
    started, process = time.monotonic(), None
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            row["pid"] = process.pid
            write_json(output / "pipeline.json", pipeline)
            next_sample = started
            while process.poll() is None:
                if time.monotonic() >= next_sample:
                    sample_gpu(resources, row, started)
                    write_json(output / "pipeline.json", pipeline)
                    next_sample = time.monotonic()+10.
                time.sleep(.25)
            row["returncode"] = process.wait()
            sample_gpu(resources, row, started)
            row["status"] = "process_complete" if process.returncode == 0 else "failed"
    except BaseException as error:
        if process is not None:
            stop_process(process, row)
            row["returncode"] = process.returncode
        else:
            row["returncode"] = None
        row.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, PipelineInterrupted)) else "failed",
                   error=f"{type(error).__name__}: {error}")
        raise
    finally:
        row.update(finished_at=timestamp(), wall_seconds=time.monotonic()-started)
        write_json(output / "pipeline.json", pipeline)
    print(f"G1_EVALUATION stage={row['name']} returncode={row['returncode']}", flush=True)
    if row["returncode"] != 0:
        raise RuntimeError(f"Stage {row['name']} failed with actual exit code {row['returncode']}; see {log_path}")


def inspect_result(destination, method, config, selector_hash):
    path = destination / "result.json"
    summary = {"result_exists": path.is_file(), "completed": False}
    if not path.is_file():
        return summary
    result = read_json(path)
    process_path = destination / "process_result.json"
    process = read_json(process_path) if process_path.is_file() else None
    summary.update(result_sha256=sha256(path), reported_completed=result.get("completed"),
        requested_episodes=result.get("requested_episodes"), counted_episodes=result.get("counted_episodes"),
        process_result=process, success_rate=result.get("success_rate"),
        recorded_task_failures=sum(bool(row.get("failure")) for row in result.get("episodes", [])),
        model_identity=result.get("model_identity"), reported_torch_gpu_peak=result.get("gpu_peak"))
    expected = set(range(config["seed_base"], config["seed_base"]+config["repeats"]))
    episodes = result.get("episodes", [])
    summary["completed"] = bool(result.get("completed") is True and result.get("method") == method
        and result.get("mode") == "closed_loop" and result.get("requested_episodes") == config["repeats"]
        and result.get("counted_episodes") == config["repeats"] and len(episodes) == config["repeats"]
        and {row.get("seed") for row in episodes} == expected
        and process is not None and process.get("returncode") == 0 and process.get("result_present") is True)
    if method == "selector" and result.get("model_identity", {}).get("checkpoint_sha256") != selector_hash:
        summary["completed"] = False
        summary["identity_error"] = "Actual selector checkpoint hash differs from the pinned input"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("isaac-python", "sonic", "reference", "asset", "selector", "embodiedskills-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        setattr(args, name, value.expanduser().resolve())
    config_path = PROJECT / "configs/evaluation.json"
    config = validate_config(read_json(config_path))
    if shutil.which("nvidia-smi") is None:
        raise FileNotFoundError("nvidia-smi is required to record actual GPU resources")
    files = {**source_files(args, config_path), **input_files(args)}
    before = fingerprint(files)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [{"name": method, "status": "pending", "output": str(args.output/method)} for method in METHODS]
    pipeline = {"schema": "g1_embodiedskills.paired_online_pipeline.v1", "started_at": timestamp(),
        "completed": False, "status": "running", "configuration": config,
        "configuration_path": str(config_path), "configuration_sha256": sha256(config_path),
        "inputs_and_sources": before, "python": {"executable": sys.executable, "version": sys.version},
        "arguments": {key: str(value) for key, value in vars(args).items()}, "methods": rows,
        "resource_log": "resources.jsonl", "resource_interval_seconds": 10,
        "resource_scope": "nvidia-smi whole-GPU memory/utilization; sampled maxima include RTX and unrelated processes",
        "baseline_model_loading": "Direct/random omit the selector bundle and report null comparison scores",
        "failure_policy": "Preserve task failures; stop on process/integrity failure. No retries or replacement seeds",
        "scope": "30 paired real online episodes per method; engineering acceptance, no statistical improvement claim"}
    started = time.monotonic()
    write_json(args.output / "pipeline.json", pipeline)

    def interrupt(signum, _frame):
        raise PipelineInterrupted(f"Parent received signal {signum}")

    previous_handlers = {signum: signal.signal(signum, interrupt) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (args.output / "resources.jsonl").open("x", encoding="utf-8") as resources:
            for row in rows:
                method = row["name"]
                destination = args.output / method
                try:
                    run_process(command_for(args, config, method, destination), args.output/f"{method}.log",
                                row, pipeline, args.output, resources)
                finally:
                    try:
                        row["result"] = inspect_result(destination, method, config,
                            before["files"]["selector_checkpoint"]["sha256"])
                    except (OSError, ValueError, TypeError, AttributeError) as error:
                        row["result"] = {"completed": False, "error": f"{type(error).__name__}: {error}"}
                    write_json(args.output / "pipeline.json", pipeline)
                changes = changed_inputs(before, {**source_files(args, config_path), **input_files(args)})
                row["input_changes"] = changes
                if changes:
                    row["status"] = "failed"
                    raise RuntimeError("Pinned sources/models/reference inputs changed during evaluation; stopping")
                if not row["result"]["completed"]:
                    row["status"] = "failed"
                    raise RuntimeError(f"Method {method} lacks a complete, valid 30-episode result; stopping without retries")
                row["status"] = "complete"
                write_json(args.output / "pipeline.json", pipeline)
            summary = {"name": "paired_audit", "status": "pending"}
            pipeline["summary"] = summary
            command = [sys.executable, str(PROJECT / "scripts/summarize_evaluation.py"),
                "--direct", str(args.output/"direct"), "--random", str(args.output/"random"),
                "--selector", str(args.output/"selector"), "--output", str(args.output/"paired-report.json")]
            run_process(command, args.output/"summary.log", summary, pipeline, args.output, resources)
            report_path = args.output / "paired-report.json"
            if not report_path.is_file():
                raise RuntimeError("Paired audit returned without its required report")
            summary.update(status="complete", report=str(report_path), report_sha256=sha256(report_path))
            changes = changed_inputs(before, {**source_files(args, config_path), **input_files(args)})
            pipeline["final_input_changes"] = changes
            if changes:
                raise RuntimeError("Pinned inputs changed before the paired audit completed")
        pipeline.update(completed=True, status="complete")
    except BaseException as error:
        pipeline.update(completed=False,
            status="interrupted" if isinstance(error, (KeyboardInterrupt, PipelineInterrupted)) else "failed",
            error={"type": type(error).__name__, "message": str(error)})
        for row in rows:
            if row["status"] == "pending":
                row["status"] = "not_started_due_to_pipeline_stop"
        raise
    finally:
        pipeline.update(finished_at=timestamp(), wall_seconds=time.monotonic()-started)
        write_json(args.output / "pipeline.json", pipeline)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    print(json.dumps({"completed": True, "pipeline": str(args.output/"pipeline.json"),
        "paired_report": str(args.output/"paired-report.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f"Online evaluation pipeline failed: {error}") from error
