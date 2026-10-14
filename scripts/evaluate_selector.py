#!/usr/bin/env python3
"""Evaluate actual archived branches with the standard EmbodiedSkills selector.

All rates in this script describe a single archived intervention followed by
its recorded continuation. They are not online, repeated-decision success
rates. No training, search, model initialization fallback, or simulator runs
are performed. Formal online evaluation seeds (>=900000) are excluded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g1_embodiedskills.io import sha256, write_json
from g1_embodiedskills.learning import read_archive, use_upstream, validate_archive

SCHEMA = "g1_embodiedskills.offline_branch_evaluation.v1"
INPUTS = ("visual", "state", "language", "progress", "actions", "mask")


def summary(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return {"count": len(values), "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None}


def conditional(values, mask):
    values, mask = np.asarray(values), np.asarray(mask, dtype=bool)
    return {"count": int(mask.sum()), "value": float(values[mask].mean()) if mask.any() else None}


def branch_metrics(success, returns, expert, selected):
    """Summarize fixed measured labels after inference, accepting tied maxima."""
    success, returns, expert, selected = (np.asarray(value) for value in (success, returns, expert, selected))
    if success.ndim != 2 or success.shape[1] != 5 or returns.shape != success.shape or expert.shape != success.shape:
        raise ValueError("Measured success/return/expert labels must be matching [N,5]")
    n = len(success)
    if not n or selected.shape != (n,) or not np.issubdtype(selected.dtype, np.integer) or np.any((selected < 0) | (selected >= 5)):
        raise ValueError("selected must contain one valid candidate index per nonempty row")
    rows = np.arange(n)
    chosen_success, direct = success[rows, selected], success[:, 0]
    oracle = success.max(axis=1)
    informative = success.max(axis=1) != success.min(axis=1)
    rescuable = (direct == 0) & (oracle == 1)
    harmable = (direct == 1) & (success.min(axis=1) == 0)
    selected_return, direct_return = returns[rows, selected], returns[:, 0]
    return_oracle = returns.max(axis=1)
    selected_expert = expert[rows, selected]
    expert_best = np.isclose(selected_expert, expert.max(axis=1), atol=1e-8, rtol=0)
    report = {
        "rows": n, "informative_rows": int(informative.sum()), "rescuable_rows": int(rescuable.sum()),
        "harmable_rows": int(harmable.sum()), "all_success_rows": int((success.min(axis=1) == 1).sum()),
        "all_failure_rows": int((oracle == 0).sum()),
        "branch_success": {"selected": float(chosen_success.mean()), "direct": float(direct.mean()),
            "uniform_random_expectation": float(success.mean()), "oracle_five_candidate_upper_bound": float(oracle.mean())},
        "conditional_accuracy": {
            "informative_selects_success": conditional(chosen_success, informative),
            "rescuable_recovery": conditional(chosen_success, rescuable),
            "harmable_preserves_direct_success": conditional(chosen_success, harmable),
            "all_direct_success_avoids_harm": conditional(chosen_success, direct == 1)},
        "rescues": int(((direct == 0) & (chosen_success == 1)).sum()),
        "harms": int(((direct == 1) & (chosen_success == 0)).sum()),
        "non_direct_selections": int((selected != 0).sum()),
        "selected_histogram": np.bincount(selected, minlength=5).tolist(),
        "physical_return": {"selected": summary(selected_return), "direct": summary(direct_return),
            "uniform_random_expectation": summary(returns.mean(axis=1)), "oracle": summary(return_oracle)},
        "return_regret": {"selected": summary(return_oracle-selected_return),
            "direct": summary(return_oracle-direct_return),
            "uniform_random_expectation": summary(return_oracle-returns.mean(axis=1)),
            "selected_on_informative": summary((return_oracle-selected_return)[informative])},
        "proxy_expert_alignment": {"tie_aware_top_utility_fraction": float(expert_best.mean()),
            "selected_utility": summary(selected_expert), "utility_regret": summary(expert.max(axis=1)-selected_expert),
            "definition": "Recorded normalized distance to frozen direct planner, not an optimal-success expert"},
        "definitions": {
            "informative": "At least one recorded successful and one unsuccessful candidate in the row",
            "rescuable": "Direct branch fails and at least one alternative branch succeeds",
            "harmable": "Direct branch succeeds and at least one alternative branch fails",
            "conditional_accuracy": "Any recorded successful choice is correct; successful ties are not reduced to one index",
            "uniform_random": "Exact equal-weight expectation over the five recorded branches; no random trials",
            "oracle": "Hindsight bound over available candidates; success and maximum return are separate bounds",
            "return_regret": "Maximum measured physical return in row minus chosen/expected measured return"},
    }
    arrays = {"success_selected": chosen_success, "success_direct": direct, "success_oracle": oracle,
        "success_uniform_random_expectation": success.mean(axis=1), "informative": informative, "rescuable": rescuable,
        "harmable": harmable, "return_selected": selected_return, "return_direct": direct_return,
        "return_oracle": return_oracle, "return_uniform_random_expectation": returns.mean(axis=1),
        "return_regret_selected": return_oracle-selected_return, "proxy_expert_top_utility": expert_best}
    return report, arrays


def _cosine(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return np.sum(left*right, axis=-1) / np.maximum(np.linalg.norm(left, axis=-1)*np.linalg.norm(right, axis=-1), 1e-12)


def evaluate(selector_path, input_path, output_dir, *, upstream_root=None, device="cpu", batch_size=32):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    selector_path, input_path, output_dir = Path(selector_path), Path(input_path), Path(output_dir)
    if not selector_path.is_file() or not input_path.is_file():
        raise FileNotFoundError("Both selector checkpoint and actual feature archive must exist")
    if output_dir.exists():
        raise FileExistsError(f"Use a fresh evaluation output directory: {output_dir}")
    data = read_archive(input_path)
    dimensions = validate_archive(data)
    if np.any(data["seed"] >= 900000):
        raise ValueError("Formal online held-out seeds >=900000 are excluded from this offline diagnostic")
    sidecar = input_path.with_suffix(".meta.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    if metadata.get("split") == "train":
        raise ValueError("Use the actual validation/test archive, not the declared training split")
    package = use_upstream(upstream_root)
    from embodiedskills.runtime import Selector, member_forward
    selector = Selector(selector_path, device=device)
    if selector.family != "continuous" or len(selector.members) != 3:
        raise ValueError("Expected the real continuous bundle with three WAMs and comparison head")
    for member in selector.payload["members"]:
        config = member["config"]
        for key in ("state_dim", "action_dim", "execution_horizon"):
            if int(config[key]) != dimensions[key]:
                raise ValueError(f"Actual archive {key} differs from checkpoint: {dimensions[key]} vs {config[key]}")
    torch.set_num_threads(1)
    scores, selected, member_scores, probabilities, future_cosines, identity_cosines = [], [], [], [], [], []
    prediction_seconds = diagnostics_seconds = 0.
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, dimensions["rows"], batch_size):
            stop = min(start+batch_size, dimensions["rows"])
            # This explicit whitelist is the entire prediction boundary. No
            # success/return/future_visual/expert label reaches either model.
            inputs = {key: data[key][start:stop] for key in INPUTS}
            before = time.perf_counter()
            prediction = selector.predict(**inputs)
            prediction_seconds += time.perf_counter()-before
            batch_scores, choices = np.asarray(prediction["scores"]), np.asarray(prediction["selected"])
            if batch_scores.shape != (stop-start, 5) or not np.isfinite(batch_scores).all():
                raise ValueError("Selector scores must be finite [B,5]")
            if choices.shape != (stop-start,) or not np.array_equal(choices, batch_scores.argmax(axis=1)):
                raise ValueError("Actual Selector choice must equal its comparison-head argmax")
            scores.append(batch_scores)
            selected.append(choices.astype(np.int64))
            before = time.perf_counter()
            tensors = {key: torch.as_tensor(value, dtype=torch.float32, device=selector.device) for key, value in inputs.items()}
            outputs = [member_forward(model, selector.family, payload["statistics"], tensors)
                       for model, payload in zip(selector.members, selector.payload["members"], strict=True)]
            member_scores.append(np.stack([out["score"].cpu().numpy() for out in outputs], axis=1))
            probabilities.append(np.stack([out["terminal_value_logit"].sigmoid().cpu().numpy() for out in outputs], axis=1))
            futures = np.stack([out["predicted_future"].cpu().numpy() for out in outputs], axis=1)
            # Label access starts only after both real inference calls finish.
            actual_future = data["future_visual"][start:stop]
            future_cosines.append(_cosine(futures, actual_future[:, None]))
            identity_cosines.append(_cosine(data["visual"][start:stop, None], actual_future))
            diagnostics_seconds += time.perf_counter()-before
    scores, selected = np.concatenate(scores), np.concatenate(selected)
    member_scores, probabilities = np.concatenate(member_scores), np.concatenate(probabilities)
    future_cosines, identity_cosines = np.concatenate(future_cosines), np.concatenate(identity_cosines)
    gain = future_cosines-identity_cosines[:, None]
    metrics, arrays = branch_metrics(data["success"], data["return"], data["expert_utility"], selected)
    members = []
    for index in range(3):
        choices = member_scores[:, index].argmax(axis=1)
        selected_metrics, _ = branch_metrics(data["success"], data["return"], data["expert_utility"], choices)
        members.append({"member": index, "config": selector.payload["members"][index]["config"],
            "future_cosine": summary(future_cosines[:, index]), "identity_cosine": summary(identity_cosines),
            "future_cosine_gain_over_identity": summary(gain[:, index]),
            "cosine_gain_per_candidate_mean": gain[:, index].mean(axis=0).tolist(),
            "cosine_gain_head_selected": summary(gain[np.arange(len(selected)), index, selected]),
            "branch_success": selected_metrics["branch_success"],
            "conditional_accuracy": selected_metrics["conditional_accuracy"],
            "proxy_expert_alignment": selected_metrics["proxy_expert_alignment"],
            "success_probability_brier": float(np.mean((probabilities[:, index]-data["success"])**2))})
    selector_hash, input_hash = sha256(selector_path), sha256(input_path)
    arrays.update({"schema": np.asarray(SCHEMA), "selector_sha256": np.asarray(selector_hash), "input_sha256": np.asarray(input_hash),
        "task": data["task"], "seed": data["seed"], "anchor": data["anchor"],
        "scores": scores, "selected": selected, "success": data["success"], "return": data["return"],
        "expert_utility": data["expert_utility"], "member_scores": member_scores,
        "member_selected": member_scores.argmax(axis=2), "member_success_probability": probabilities,
        "member_future_cosine": future_cosines, "identity_future_cosine": identity_cosines,
        "member_future_cosine_gain": gain})
    output_dir.mkdir(parents=True, exist_ok=False)
    archive = output_dir / "predictions.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": SCHEMA, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Offline single archived branch selection with recorded continuation; not online closed-loop success rate",
        "input": str(input_path.resolve()), "input_sha256": input_hash,
        "input_sidecar_sha256": sha256(sidecar) if sidecar.is_file() else None,
        "declared_split": metadata.get("split", "not_declared"), "dimensions": dimensions,
        "selector": str(selector_path.resolve()), "selector_sha256": selector_hash,
        "upstream_source_hashes": {name: sha256(package/name) for name in ("runtime.py", "checkpoint.py", "continuous.py", "continuous_consensus.py")},
        "model": {"family": selector.family, "members": 3, "comparison_head": type(selector.head).__name__,
            "hidden_sizes": [member["config"].get("hidden", 192) for member in selector.payload["members"]],
            "encoders": selector.payload.get("encoders", {})},
        "prediction_inputs": list(INPUTS), "labels_enter_prediction": False,
        "metrics": metrics, "member_diagnostics": members,
        "future_diagnostic_definition": "cos(predicted_future,recorded_future)-cos(current_visual,recorded_future), all five candidate continuations",
        "predictions_archive": archive.name, "predictions_archive_sha256": sha256(archive),
        "device": str(device), "batch_size": batch_size, "prediction_seconds": prediction_seconds,
        "member_diagnostic_seconds": diagnostics_seconds, "wall_seconds": time.monotonic()-started,
        "limitations": ["Branches share archived anchors; no repeated online interventions were evaluated",
            "Uniform-random value is an exact archive expectation; oracle uses measured future labels only as a bound",
            "Proxy expert labels measure planner distance, not optimal task success",
            "Rows within the same task/seed are correlated; no independence-based confidence interval is claimed"]}
    write_json(output_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory for report.json and predictions.npz")
    parser.add_argument("--embodiedskills-root", type=Path, default=os.environ.get("EMBODIEDSKILLS_ROOT"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = evaluate(args.selector, args.input, args.output, upstream_root=args.embodiedskills_root,
                      device=args.device, batch_size=args.batch_size)
    print(json.dumps({"scope": report["scope"], "branch_success": report["metrics"]["branch_success"],
                      "report": str(args.output/"report.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
