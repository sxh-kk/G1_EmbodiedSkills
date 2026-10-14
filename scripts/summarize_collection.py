"""Validate measured branch identities and combine accepted raw archives."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g1_embodiedskills.collection import _native_digest, prefix_comparison, reference_utility
from g1_embodiedskills.io import write_json, sha256
from g1_embodiedskills.learning import validate_archive
from g1_embodiedskills.metrics import episode_outcome


@lru_cache(maxsize=5)
def load_trace(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-object-gate", action="store_true", help="Require full cube pose/velocity replay checks")
    args = parser.parse_args()
    archives, sources, identities = [], [], set()
    maximum_prefix = {}
    common_provenance = None
    for directory in args.collections:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if args.require_object_gate and "cube_root" not in manifest.get("physical_prefix_fields", []):
            raise ValueError("Collection predates the complete physical-object replay gate")
        provenance = {key: value for key, value in manifest.get("provenance", {}).items()
            if key != "collection_config_sha256"}
        if common_provenance is None:
            common_provenance = provenance
        elif provenance != common_provenance:
            raise ValueError("Cannot combine collections with different source, assets or references")
        with np.load(directory / "raw.npz", allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        validate_archive(data, raw=True)
        groups = {(int(group["seed"]), int(group["anchor_step"])): group
            for group in manifest["groups"] if group["accepted"]}
        group_paths = {}
        for batch in sorted(directory.glob("batch-*")):
            for group in json.loads((batch / "groups-in-progress.json").read_text(encoding="utf-8")):
                if not group["rejections"]:
                    group_paths[(int(group["seed"]), int(group["anchor_step"]))] = batch
        for row, (task, seed, anchor) in enumerate(zip(data["task"], data["seed"], data["anchor"])):
            identity = (str(task), int(seed), int(anchor))
            if identity in identities:
                raise ValueError(f"Duplicate episode/anchor across collections: {identity}")
            identities.add(identity)
            group = groups[(int(seed), int(anchor))]
            if group["rejections"] or len(group["runs"]) != 5:
                raise ValueError("Rejected/incomplete group found in training archive")
            for run in group["runs"]:
                cid = run["candidate"]
                if not run["native_executed"] or not run.get("canonical_encoding_bitwise_equal"):
                    raise ValueError("Unexecuted or mismatched candidate in archive")
                if run["native_sha256"] != run["canonical_native_sha256"]:
                    raise ValueError("Canonical/native command hash mismatch")
                native_path = group_paths[(int(seed), int(anchor))] / f"candidate-{cid}" / f"native-slot-{group['slot']}.npz"
                if args.require_object_gate:
                    prefix_name = f"prefix-slot-{group['slot']}.npz"
                    baseline_path = native_path.parent.parent / "candidate-0" / prefix_name
                    with np.load(baseline_path, allow_pickle=False) as source:
                        baseline = {key: source[key] for key in source.files}
                    with np.load(native_path.parent / prefix_name, allow_pickle=False) as source:
                        actual_prefix = {key: source[key] for key in source.files}
                    if "cube_root" not in baseline or not prefix_comparison(baseline, actual_prefix,
                            manifest["prefix_tolerance"])["passed"]:
                        raise ValueError("Saved physical/object/history prefixes fail recomputed replay gate")
                with np.load(native_path, allow_pickle=False) as source:
                    native = tuple(source[name] for name in ("qtraj_mjcf", "roots", "root_quaternion_wxyz", "finger_joint_targets"))
                if _native_digest(native) != run["native_sha256"]:
                    raise ValueError("Saved executable differs from the recorded native command")
                encoded = np.concatenate(native, axis=-1).astype(np.float32)
                if not np.array_equal(encoded, data["actions"][row, cid]):
                    raise ValueError("Saved executable differs from the actual training action")
                trace = load_trace(native_path.parent / "trace.npz")
                slot = group["slot"]
                origin = trace.get("env_origins")
                if origin is None:
                    # Early plane-scene pilots predate explicit origin fields;
                    # their local reset root XY is exactly zero and floor Z=0.
                    origin = np.r_[trace["root"][0, slot, :2], 0.]
                else:
                    origin = origin[slot]
                terminated = bool(trace["terminated"][:,slot].any()) if "terminated" in trace else run["outcome"]["terminated"]
                measured = episode_outcome(trace, slot=slot, origin=origin, terminated=terminated)
                if measured["success"] != run["outcome"]["success"] or not np.isclose(
                    measured["return_value"], run["outcome"]["return_value"], atol=1e-6, rtol=1e-6):
                    raise ValueError("Recorded outcome differs from recomputed physical trace")
                for comparison in (run.get("initial_comparison"), run.get("prefix_comparison")):
                    if comparison is None:
                        continue
                    if not comparison["passed"]:
                        raise ValueError("Failed physical prefix gate in training archive")
                    for name, metric in comparison["fields"].items():
                        maximum_prefix[name] = max(maximum_prefix.get(name, 0.), metric["max_abs"])
                if data["success"][row, cid] != float(run["outcome"]["success"]):
                    raise ValueError("Success label differs from measured outcome")
                if not np.isclose(data["return"][row, cid], run["outcome"]["return_value"], atol=1e-6, rtol=1e-6):
                    raise ValueError("Return label differs from measured outcome")
                utility = reference_utility(data["actions"][row, cid], data["actions"][row, 0])
                if not np.isclose(data["expert_utility"][row, cid], utility, atol=1e-8, rtol=1e-6):
                    raise ValueError("Utility differs from actual native-action distance")
        sources.append(dict(anchors=len(data["task"]), raw_sha256=sha256(directory / "raw.npz"),
            collection_config_sha256=manifest.get("provenance", {}).get("collection_config_sha256"),
            attempted_groups=manifest["attempted_groups"], rejected_groups=manifest["rejected_groups"],
            jitter_m=manifest["position_jitter_m"], wall_seconds=manifest["wall_seconds"]))
        archives.append(data)
    combined = {key: np.concatenate([data[key] for data in archives], axis=0) for key in archives[0]}
    validate_archive(combined, raw=True)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    temporary = args.output.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **combined)
    temporary.replace(args.output)
    success = combined["success"]
    summary = dict(anchors=len(success), branches=success.size,
        unique_episodes=len(set(zip(combined["task"].tolist(), combined["seed"].tolist()))),
        successes=int(success.sum()), failures=int(success.size-success.sum()),
        direct_success_rate=float(success[:, 0].mean()),
        any_candidate_success_rate=float(success.max(1).mean()),
        informative_anchors=int((success.max(1)>success.min(1)).sum()),
        rescuable_anchors=int(((success[:,0]==0)&(success.max(1)>0)).sum()),
        harmful_alternatives_anchors=int(((success[:,0]>0)&(success.min(1)==0)).sum()),
        candidate_success_counts=success.sum(0).astype(int).tolist(),
        state_dim=int(combined["state"].shape[1]), action_shape=list(combined["actions"].shape[1:]),
        prefix_maximum_absolute_errors=maximum_prefix,
        labels_verified_against_recorded_physical_outcomes=True,
        outcomes_recomputed_from_physical_traces=True,
        canonical_command_hashes_verified=True,
        physical_object_prefix_recomputed=args.require_object_gate,
        provenance=common_provenance,
        output_sha256=sha256(args.output), sources=sources)
    write_json(args.report, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
