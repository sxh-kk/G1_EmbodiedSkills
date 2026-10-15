#!/usr/bin/env python3
"""Recompute M1 comparison scores from the recorded deployment inputs.

This is a prediction-integrity audit, not an outcome analysis. It neither
simulates episodes nor prints success counts. Run it after the sealed suite has
finished; physical outcomes and paired starts are audited separately.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from g1_embodiedskills.io import sha256, write_json
from g1_embodiedskills.m1_runtime import SELECTOR_KINDS

ATOL, RTOL = 1e-5, 1e-6
SHAPES = {"visual": (1024,), "state": (1041,), "progress": (1,),
          "actions": (5, 75, 38), "mask": (5, 75), "scores": (5,)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON value {token} in {path}")))


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def inside(directory, relative):
    require(isinstance(relative, str) and bool(relative) and not Path(relative).is_absolute(),
            "Recorded inference/archive paths must be nonempty relative paths")
    result = (directory / relative).resolve()
    require(result.is_relative_to(directory) and result.is_file(),
            f"Missing or escaping recorded archive: {relative}")
    return result


def integer(value, label, low=0, high=None):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
            f"{label}: expected an integer")
    value = int(value)
    require(value >= low and (high is None or value <= high), f"{label}: outside allowed range")
    return value


def array(value, shape, label):
    result = np.asarray(value)
    require(result.shape == shape and result.dtype == np.float32 and np.isfinite(result).all(),
            f"{label}: expected finite float32 array of shape {shape}")
    return result


def prediction_check(actual, recorded, selected, *, label):
    """Keep score tolerance and exact decision agreement independent."""
    actual, recorded = np.asarray(actual), np.asarray(recorded)
    require(actual.shape == recorded.shape and actual.ndim == 2 and actual.shape[1] == 5
            and actual.dtype == recorded.dtype == np.float32
            and np.isfinite(actual).all() and np.isfinite(recorded).all(),
            f"{label}: malformed recomputed or recorded scores")
    selected = np.asarray(selected)
    require(selected.shape == (len(actual),) and np.issubdtype(selected.dtype, np.integer),
            f"{label}: malformed selected indices")
    difference = np.abs(actual.astype(np.float64)-recorded.astype(np.float64))
    maximum = float(difference.max()) if difference.size else 0.
    require(np.allclose(actual, recorded, atol=ATOL, rtol=RTOL),
            f"{label}: recomputed score mismatch; max_abs={maximum:.9g}, atol={ATOL}, rtol={RTOL}")
    require(np.array_equal(actual.argmax(axis=1), selected)
            and np.array_equal(recorded.argmax(axis=1), selected),
            f"{label}: recomputed/recorded argmax differs from the actual selected candidate")
    return maximum, float(np.square(difference).sum()), int(difference.size)


@contextmanager
def fp32_prediction(torch):
    """Match online's local FP32 matmul mode without changing SONIC state."""
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def tensor_fingerprint(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        # uint8 supports every storage dtype, including bfloat16 if introduced.
        digest.update(tensor.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return {"class": type(module).__name__, "state_sha256": digest.hexdigest(),
            "parameters": sum(parameter.numel() for parameter in module.parameters())}


def fingerprint(files):
    result = {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
              for path in sorted(files)}
    combined = json.dumps({name: row["sha256"] for name, row in result.items()}, sort_keys=True).encode()
    return {"files": result, "combined_sha256": hashlib.sha256(combined).hexdigest()}


def source_files():
    result = {Path(__file__).resolve()}
    for name, module in list(sys.modules.items()):
        if name.startswith(("g1_embodiedskills", "embodiedskills")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if path.suffix == ".py" and path.is_file():
                result.add(path)
    return result


def audit(directory, device, report):
    import torch
    from g1_embodiedskills.learning import use_upstream

    request_path, result_path = directory / "request.json", directory / "result.json"
    process_path = directory / "process_result.json"
    request, result, process = map(read_json, (request_path, result_path, process_path))
    require(request.get("method") == result.get("method") == "selector",
            "Prediction audit requires a learned selector run")
    require(result.get("completed") is True and process.get("returncode") == 0
            and process.get("result_present") is True, "Run did not finish normally")
    kind = request.get("selector_kind")
    require(kind in SELECTOR_KINDS, f"Unknown selector kind: {kind}")
    identity = result.get("model_identity", {})
    require(identity.get("selector_kind") == kind, "Requested and loaded selector kinds differ")
    checkpoint = Path(request["selector"]).expanduser().resolve()
    require(checkpoint.is_file() and Path(identity["checkpoint"]).resolve() == checkpoint,
            "Requested and loaded checkpoint paths differ")
    checkpoint_hash = sha256(checkpoint)
    require(identity.get("checkpoint_sha256") == checkpoint_hash, "Checkpoint changed after online execution")
    context_path = inside(directory, identity.get("inference_context_archive"))
    require(sha256(context_path) == identity.get("inference_context_sha256"),
            "Recorded inference context hash differs")
    language = array(read_npz(context_path).get("language_vector"), (384,), "language_vector")
    require(hashlib.sha256(language.tobytes()).hexdigest() == identity.get("language_vector_sha256"),
            "Recorded language vector differs from the actual online vector hash")
    require(np.isclose(np.linalg.norm(language), 1., atol=5e-5, rtol=0), "Language embedding is not L2 normalized")
    count = integer(request.get("num_envs"), "num_envs", low=1)
    repeats = integer(request.get("repeats"), "repeats", low=1)
    seed_base = integer(request.get("seed_base"), "seed_base", low=900000)
    total_steps = round(float(request["seconds"])/.02)
    require(total_steps > 0 and total_steps % 10 == 0, "Invalid control budget")
    require(result.get("counted_episodes") == result.get("requested_episodes") == repeats,
            "Requested and completed episode counts differ")
    batches = sorted(path for path in directory.glob("batch-*") if path.is_dir())
    require([path.name for path in batches] == [f"batch-{index:04d}" for index in range((repeats+count-1)//count)],
            "Missing or unexpected batch directories")

    # Load every recorded row and freeze the exact evidence inventory before inference.
    files = {request_path, result_path, process_path, context_path, checkpoint}
    grouped = []
    for batch_index, batch in enumerate(batches):
        rows_path = batch / "decisions.json"
        rows = read_json(rows_path)
        files.add(rows_path)
        require(isinstance(rows, list), f"{batch.name}: decisions must be an array")
        groups, seen = {}, set()
        active_count = min(count, repeats-batch_index*count)
        for row in rows:
            require(isinstance(row, dict), f"{batch.name}: malformed decision row")
            slot = integer(row.get("slot"), "slot", high=active_count-1)
            step = integer(row.get("step"), "step", high=total_steps-10)
            require(step % 10 == 0 and (step, slot) not in seen,
                    f"{batch.name}: repeated or off-clock decision")
            seen.add((step, slot))
            require(row.get("seed") == seed_base+batch_index*count+slot,
                    f"{batch.name}: seed/slot identity differs")
            path = inside(directory, row.get("audit_archive"))
            files.add(path)
            if row.get("status") == "proposal_failed":
                require(row.get("selected") is None and row.get("scores") is None,
                        f"{batch.name}: failed proposal claims model selection")
                continue
            require(row.get("status") == "selected" and row.get("method") == "selector",
                    f"{batch.name}: invalid learned-selection record")
            require(row.get("selector_kind") == kind and row.get("checkpoint_sha256") == checkpoint_hash,
                    f"{batch.name}: decision model identity differs from the loaded model")
            groups.setdefault(step, []).append((slot, row, path))
        for step in sorted(groups):
            group = groups[step]
            require([item[0] for item in group] == sorted(item[0] for item in group),
                    f"{batch.name}/step={step}: recorded inference slots are not in runtime order")
            grouped.append((batch.name, step, group))

    use_upstream(request.get("upstream_root"))
    torch.set_num_threads(1)
    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA prediction audit requested but CUDA is unavailable")
    if kind == "legacy":
        from embodiedskills.runtime import Selector
        selector = Selector(checkpoint, device=str(selected_device))
        require(selector.payload.get("m1_variant") is None, "Legacy run loaded an M1 checkpoint")
    else:
        from g1_embodiedskills.m1_learning import load_m1_selector
        selector = load_m1_selector(checkpoint, device=str(selected_device))
        require(selector.payload.get("m1_variant") == kind, "Checkpoint M1 variant differs from the run kind")
    require(selector.family == "continuous" and len(selector.members) == 3 and selector.head is not None,
            "Prediction audit requires the actual three-member comparison model")
    modules = [*selector.members, selector.head]
    require(all(not module.training for module in modules), "Loaded prediction modules are not in eval mode")
    module_records = [tensor_fingerprint(module) for module in modules]
    report.update(selector_kind=kind, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                  device=str(selected_device), torch_version=torch.__version__,
                  precision="FP32; matmul TF32 disabled locally, as in online inference",
                  tolerances={"absolute": ATOL, "relative": RTOL, "argmax": "exact"},
                  model_modules={"members": module_records[:3], "comparison_head": module_records[3]},
                  model_parameter_count=sum(row["parameters"] for row in module_records),
                  language_vector_sha256=identity["language_vector_sha256"],
                  recorded_runtime_audit_contract=result.get("runtime_audit_contract"))
    before = fingerprint(files)
    sources = source_files()
    source_before = fingerprint(sources)
    report["recorded_input_fingerprint"] = before
    report["inference_source_fingerprint"] = source_before
    checked, maximum, square_sum, elements = 0, 0., 0., 0
    group_reports = []
    for batch_name, step, group in grouped:
        label = f"{batch_name}/step={step}"
        inputs = {key: [] for key in SHAPES}
        selected = []
        for slot, row, path in group:
            data = read_npz(path)
            for key, shape in SHAPES.items():
                inputs[key].append(array(data.get(key), shape, f"{label}/slot={slot}/{key}"))
            require(np.array_equal(np.asarray(row["scores"], dtype=np.float64),
                                   data["scores"].astype(np.float64)),
                    f"{label}/slot={slot}: JSON/NPZ scores differ")
            index = integer(row.get("selected"), "selected", high=4)
            saved_index = np.asarray(data.get("selected"))
            require(saved_index.shape == () and saved_index.dtype.kind in "iu" and int(saved_index) == index
                    and row.get("head_argmax") == index,
                    f"{label}/slot={slot}: actual selected/head/archive indices differ")
            require(np.all(data["mask"] == 1), f"{label}: online M1 mask must be all valid")
            require(np.isclose(data["progress"][0], step/total_steps, atol=1e-7, rtol=0),
                    f"{label}: recorded progress differs from the actual control budget")
            require(np.isclose(np.linalg.norm(data["visual"]), 1., atol=5e-5, rtol=0),
                    f"{label}: visual embedding is not L2 normalized")
            selected.append(index)
        inputs = {key: np.stack(value) for key, value in inputs.items()}
        recorded = inputs.pop("scores")
        inputs["language"] = np.repeat(language[None], len(group), axis=0)
        with fp32_prediction(torch):
            prediction = selector.predict(**inputs)
        predicted_indices = np.asarray(prediction["selected"])
        require(np.array_equal(predicted_indices, np.asarray(selected, dtype=np.int64)),
                f"{label}: reloaded model selected a different candidate")
        error, square, size = prediction_check(prediction["scores"], recorded, selected, label=label)
        maximum, square_sum, elements = max(maximum, error), square_sum+square, elements+size
        checked += len(group)
        group_reports.append({"batch": batch_name, "step": step, "slots": [item[0] for item in group],
                              "decisions": len(group), "maximum_absolute_score_error": error,
                              "argmax_exact": True})
    require(result.get("decisions") == checked, "Recomputed decision count differs from runtime count")
    require(fingerprint(files) == before, "Recorded inputs/checkpoint changed during the prediction audit")
    require(fingerprint(sources) == source_before, "Inference source files changed during the prediction audit")
    report.update(status="complete", completed=True, inference_batches=len(grouped), decisions_checked=checked,
                  score_values_checked=elements, maximum_absolute_score_error=maximum,
                  score_rmse=float(np.sqrt(square_sum/elements)) if elements else None,
                  argmax_exact=True, recorded_inputs_unchanged=True, inference_sources_unchanged=True,
                  batch_groups=group_reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="One completed learned-policy run")
    parser.add_argument("--output", type=Path, required=True, help="Fresh method-level audit JSON")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    directory, output = args.directory.expanduser().resolve(), args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite an existing prediction audit: {output}")
    report = {"schema": "g1_embodiedskills.m1_prediction_replay_audit.v1",
              "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "directory": str(directory), "status": "running", "completed": False,
              "scope": "Recompute deployment scores in actual batch/step/slot groups; no outcome analysis"}
    try:
        audit(directory, args.device, report)
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise SystemExit(f"M1 prediction audit failed: {type(error).__name__}: {error}") from error
    finally:
        report["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        write_json(output, report)
    # Deliberately omit task outcomes, rates and comparative results.
    print(json.dumps({"prediction_audit": str(output), "status": report["status"],
                      "decisions_checked": report["decisions_checked"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
