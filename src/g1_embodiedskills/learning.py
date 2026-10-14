"""Offline G1 archives and unmodified upstream EmbodiedSkills training.

The raw contract is RGB/future RGB, instruction, state, elapsed-budget progress,
five encoded action references, mask, task/seed/anchor, and measured branch
success/return/expert_utility. No label or feature is invented here. All anchors
from a (task, seed) episode stay in one split. Model loading is offline only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np

IDENTITIES = ("task", "seed", "anchor")
NUMERIC = ("state", "progress", "actions", "mask", "success", "return", "expert_utility")
FEATURES = ("visual", "future_visual", "language")
DEFAULT_VISUAL = "facebook/vjepa2-vitl-fpc64-256"
DEFAULT_LANGUAGE = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VISUAL_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
DEFAULT_LANGUAGE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_CONFIG = {
    "family": "continuous", "seed": 0, "member_seeds": [0, 1, 2], "batch_size": 128,
    "model": {
        "hidden": 192, "success_priority": 4.0, "outcome_change_standardized": True,
        "detach_consequence_for_score": True,
    },
    "member": {"epochs": 30, "learning_rate": 0.0003, "expert_weight": 0.5},
    "consensus": {
        "epochs": 40, "batch_size": 128, "samples_per_task": 256, "learning_rate": 0.0003,
    },
    "encoders": {
        "visual_backbone": "vjepa2", "visual_model": DEFAULT_VISUAL,
        "visual_revision": DEFAULT_VISUAL_REVISION,
        "language_model": DEFAULT_LANGUAGE, "language_revision": DEFAULT_LANGUAGE_REVISION,
        "spatial_grid": 1,
    },
}


def use_upstream(root: str | Path | None = None) -> Path:
    """Use an installed upstream package or EMBODIEDSKILLS_ROOT/src, without pip."""
    selected = root or os.environ.get("EMBODIEDSKILLS_ROOT")
    if selected:
        source = Path(selected).expanduser().resolve() / "src"
        if not (source / "embodiedskills" / "training.py").is_file():
            raise FileNotFoundError("EMBODIEDSKILLS_ROOT must contain src/embodiedskills/training.py")
        spec = importlib.util.find_spec("embodiedskills")
        if spec is not None and spec.origin:
            if Path(spec.origin).resolve().parent != source / "embodiedskills":
                raise ValueError("Another embodiedskills package is already visible; use one upstream source")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    try:
        package = importlib.import_module("embodiedskills")
    except ModuleNotFoundError as error:
        raise ImportError("Install EmbodiedSkills or set EMBODIEDSKILLS_ROOT to its checkout") from error
    return Path(package.__file__).resolve().parent


def read_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def validate_archive(data: dict[str, np.ndarray], *, raw: bool = False) -> dict:
    """Validate training inputs before loading a GPU or writing an output archive."""
    image_keys = ("rgb", "future_rgb", "instruction") if raw else FEATURES
    missing = set((*IDENTITIES, *NUMERIC, *image_keys)) - data.keys()
    if missing:
        raise ValueError(f"Missing real archive fields: {sorted(missing)}; labels are never synthesized")
    n = len(data["task"])
    if not n:
        raise ValueError("Archive must contain at least one anchor")
    for key in (*IDENTITIES, *NUMERIC, *image_keys):
        if data[key].ndim == 0 or len(data[key]) != n or data[key].dtype.hasobject:
            raise ValueError(f"{key} must have {n} rows without Python object dtype")
    if data["task"].shape != (n,) or data["task"].dtype.kind not in "US":
        raise ValueError("task must be a one-dimensional string array")
    if any(not str(task).strip() for task in data["task"]):
        raise ValueError("task strings must be nonempty")
    for key in ("seed", "anchor"):
        if data[key].shape != (n,) or data[key].dtype.kind not in "iu":
            raise ValueError(f"{key} must be a one-dimensional integer array")
    identities = list(zip(data["task"].astype(str), data["seed"].tolist(), data["anchor"].tolist()))
    if len(set(identities)) != n:
        raise ValueError("Duplicate task/seed/anchor identity")
    for key in NUMERIC + (() if raw else FEATURES):
        if data[key].dtype.kind not in "biuf" or not np.isfinite(data[key]).all():
            raise ValueError(f"{key} must contain finite real numeric values")
    state, actions = data["state"], data["actions"]
    if state.ndim != 2 or state.shape[1] <= 0:
        raise ValueError("state must have shape [N,S], S > 0")
    if actions.ndim != 4 or actions.shape[:2] != (n, 5) or min(actions.shape[2:]) <= 0:
        raise ValueError("actions must have shape [N,5,H,A], H/A > 0")
    if data["progress"].shape != (n, 1) or np.any((data["progress"] < 0) | (data["progress"] > 1)):
        raise ValueError("progress must be elapsed budget fraction [N,1] in [0,1]")
    mask = data["mask"]
    if mask.shape != actions.shape[:3] or not np.isin(mask, [0, 1]).all():
        raise ValueError("mask must be binary [N,5,H]")
    if np.any(mask.sum(-1) == 0) or np.any(np.diff(mask.astype(np.int8), axis=-1) > 0):
        raise ValueError("Every candidate needs a nonempty valid prefix followed only by padding")
    for key in ("success", "return", "expert_utility"):
        if data[key].shape != (n, 5):
            raise ValueError(f"{key} must have shape [N,5]")
    if not np.isin(data["success"], [0, 1]).all():
        raise ValueError("success must contain measured binary branch outcomes")
    expected = {"rgb": (n, 256, 256, 3), "future_rgb": (n, 5, 256, 256, 3)} if raw else {
        "visual": (n, 1024), "future_visual": (n, 5, 1024), "language": (n, 384),
    }
    for key, shape in expected.items():
        if data[key].shape != shape or (raw and data[key].dtype != np.uint8):
            raise ValueError(f"{key} must have shape {shape}" + (" and dtype uint8" if raw else ""))
    if raw and (
        data["instruction"].shape != (n,) or data["instruction"].dtype.kind not in "US"
        or any(not str(text).strip() for text in data["instruction"])
    ):
        raise ValueError("instruction must contain N nonempty strings")
    success = data["success"]
    return {
        "rows": n, "episodes": len(set(zip(data["task"].astype(str), data["seed"].tolist()))),
        "state_dim": int(state.shape[1]), "action_dim": int(actions.shape[-1]),
        "execution_horizon": int(actions.shape[2]), "candidates": 5,
        "branch_success_rate": float(success.mean()),
        "informative_rows": int(np.sum(success.max(1) != success.min(1))),
        "return_range": [float(data["return"].min()), float(data["return"].max())],
        "expert_utility_range": [float(data["expert_utility"].min()), float(data["expert_utility"].max())],
    }


def _digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save_archive(path: str | Path, data: dict, metadata: dict) -> None:
    path = Path(path)
    sidecar = path.with_suffix(".meta.json")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **data)
        temporary.replace(path)
        _write_json(sidecar, metadata)
    finally:
        temporary.unlink(missing_ok=True)


def _model_inventory(model: str) -> dict:
    """Inspect a model directory/HF snapshot without network or model allocation."""
    explicit = Path(model).expanduser()
    hub = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
    locations = [explicit] if explicit.is_dir() else sorted((hub / ("models--" + model.replace("/", "--")) / "snapshots").glob("*"))
    snapshots = []
    for location in locations:
        files = sorted(str(item.relative_to(location)) for item in location.rglob("*") if item.is_file())
        weights = [name for name in files if name.endswith((".safetensors", "pytorch_model.bin"))]
        missing_shards = []
        for index in location.glob("*.index.json"):
            try:
                expected = set(json.loads(index.read_text())["weight_map"].values())
                missing_shards.extend(sorted(name for name in expected if not (location / name).is_file()))
            except (ValueError, KeyError):
                missing_shards.append(f"invalid index: {index.name}")
        snapshots.append({
            "location": str(location), "config": (location / "config.json").is_file(),
            "weights": weights, "missing_shards": missing_shards,
            "tokenizer": any((location / name).is_file() for name in ("tokenizer.json", "vocab.txt", "tokenizer.model")),
        })
    return {"model": model, "snapshots": snapshots}


def encoder_inventory(visual_model: str = DEFAULT_VISUAL, language_model: str = DEFAULT_LANGUAGE) -> dict:
    packages = {}
    for name in ("numpy", "torch", "transformers", "pyyaml", "huggingface-hub", "pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"packages": packages, "visual": _model_inventory(visual_model), "language": _model_inventory(language_model), "network_access": False}


def encoder_revision(model: str | Path, revision: str | None = None) -> str | None:
    if revision is not None:
        return revision
    return {DEFAULT_VISUAL: DEFAULT_VISUAL_REVISION, DEFAULT_LANGUAGE: DEFAULT_LANGUAGE_REVISION}.get(str(model))


def resolve_offline_model(model: str | Path, revision: str | None = None) -> Path:
    """Resolve a pinned cached snapshot, never querying/downloading from the Hub.

    Upstream encoders accept a local path rather than a revision argument. A
    SHA-only download has no ``refs/main``; resolving the snapshot explicitly
    makes extraction and online encoding use the same immutable weights.
    """
    explicit = Path(model).expanduser()
    if explicit.is_dir():
        return explicit.resolve()
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(str(model), revision=encoder_revision(model, revision), local_files_only=True))
    except OSError as error:
        raise FileNotFoundError(f"Offline encoder snapshot unavailable: {model}@{encoder_revision(model, revision) or 'main'}; prepare its required model files first") from error


def validate_language_cache(cache: dict) -> dict[str, np.ndarray]:
    """Validate portable, exact float32 embeddings used offline and online."""
    if not isinstance(cache, dict) or cache.get("schema") != "g1_embodiedskills.language_cache.v1" or cache.get("encoding") != "single_instruction_batch_one":
        raise ValueError("Expected G1 single-instruction batch-one language cache")
    embeddings, hashes = cache.get("embeddings"), cache.get("sha256")
    if not isinstance(embeddings, dict) or not embeddings or not isinstance(hashes, dict) or embeddings.keys() != hashes.keys():
        raise ValueError("Language cache embeddings/hash instruction keys must match")
    vectors = {}
    for instruction, values in embeddings.items():
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Language cache instructions must be nonempty strings")
        vector = np.asarray(values, dtype=np.float32)
        if vector.shape != (384,) or not np.isfinite(vector).all():
            raise ValueError("Language cache vectors must be finite float32[384]")
        if hashlib.sha256(vector.tobytes()).hexdigest() != hashes[instruction]:
            raise ValueError(f"Language cache SHA256 mismatch for instruction: {instruction}")
        if not np.isclose(np.linalg.norm(vector), 1., atol=1e-4, rtol=0):
            raise ValueError("Language cache must contain actual L2-normalized MiniLM vectors")
        vectors[instruction] = vector
    return vectors


def build_language_cache(instructions, language_encoder) -> dict:
    """Encode each unique instruction exactly once, with a single-item batch.

    Fixed-language datasets have nearly zero language variance. Tiny batch-size
    dependent encoder differences would be amplified by upstream normalization
    and LayerNorm, so the actual frozen embedding is cached in the final bundle.
    """
    texts = list(instructions)
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Language extraction needs nonempty instruction strings")
    embeddings, hashes = {}, {}
    for instruction in sorted(set(texts)):
        encoded = np.asarray(language_encoder.encode([instruction], batch_size=1), dtype=np.float32)
        if encoded.shape != (1, 384) or not np.isfinite(encoded).all():
            raise ValueError("Actual frozen language encoder must return finite [1,384]")
        vector = np.ascontiguousarray(encoded[0])
        embeddings[instruction] = vector.tolist()
        hashes[instruction] = hashlib.sha256(vector.tobytes()).hexdigest()
    cache = {"schema": "g1_embodiedskills.language_cache.v1", "encoding": "single_instruction_batch_one", "embeddings": embeddings, "sha256": hashes}
    validate_language_cache(cache)
    return cache


def cached_language_vectors(instructions, cache: dict) -> np.ndarray:
    """Return exact cached vectors; unknown text has no implicit encoder fallback."""
    vectors = validate_language_cache(cache)
    texts = list(instructions)
    unknown = sorted(set(texts) - vectors.keys())
    if unknown:
        raise ValueError(f"Instruction absent from frozen language cache; language generalization is not supported: {unknown}")
    return np.stack([vectors[text] for text in texts]) if texts else np.empty((0, 384), dtype=np.float32)


def training_language_cache(train: dict, validation: dict, metadata: list[dict]) -> dict:
    """Require matching cache provenance and exact feature rows before training."""
    caches = [item.get("g1_language_cache") for item in metadata]
    if len(caches) != 2 or any(cache is None for cache in caches):
        raise ValueError("Train/validation feature metadata need g1_language_cache; re-extract with single-instruction batch-one encoding")
    vectors = validate_language_cache(caches[0])
    validate_language_cache(caches[1])
    if caches[0] != caches[1]:
        raise ValueError("Training and validation language caches differ")
    allowed = {vector.tobytes() for vector in vectors.values()}
    for name, data in (("train", train), ("validation", validation)):
        if any(np.asarray(row, dtype=np.float32).tobytes() not in allowed for row in data["language"]):
            raise ValueError(f"{name} language feature differs from its exact frozen cache")
    return copy.deepcopy(caches[0])


def attach_language_cache(bundle_path: str | Path, cache: dict) -> None:
    """Add exact encoder metadata without changing standard WAM bundle fields."""
    import torch
    validate_language_cache(cache)
    bundle = Path(bundle_path)
    payload = torch.load(bundle, map_location="cpu", weights_only=True)
    if payload.get("schema") != "embodiedskills.bundle.v1" or len(payload.get("members", [])) != 3:
        raise ValueError("Expected the standard three-member EmbodiedSkills bundle")
    payload["g1_language_cache"] = copy.deepcopy(cache)
    temporary = bundle.with_name(f".{bundle.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(bundle)
    finally:
        temporary.unlink(missing_ok=True)


def encode_arrays(data: dict, visual_encoder, language_encoder, *, batch_size: int = 16, language_cache: dict | None = None) -> dict:
    """Apply upstream frozen encoders and continuous-family L2 normalization."""
    validate_archive(data, raw=True)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if visual_encoder.feature_dim != 1024 or language_encoder.feature_dim != 384:
        raise ValueError("Continuous EmbodiedSkills requires visual=1024 and language=384")

    def image_features(images):
        features = np.asarray(visual_encoder.encode_rgb(images, batch_size=batch_size), dtype=np.float32)
        if features.shape != (len(images), 1, 1024):
            raise ValueError("Frozen visual encoder must return [B,1,1024] for spatial_grid=1")
        features = features[:, 0]
        return features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)

    n = len(data["task"])
    output = {key: data[key] for key in (*IDENTITIES, *NUMERIC)}
    output["visual"] = image_features(data["rgb"])
    output["future_visual"] = image_features(data["future_rgb"].reshape(-1, 256, 256, 3)).reshape(n, 5, 1024)
    instructions = data["instruction"].astype(str).tolist()
    cache = language_cache if language_cache is not None else build_language_cache(instructions, language_encoder)
    output["language"] = cached_language_vectors(instructions, cache)
    validate_archive(output)
    return output


def extract_features(input_path, output_path, *, visual_model=DEFAULT_VISUAL, language_model=DEFAULT_LANGUAGE, visual_revision=None, language_revision=None, device="cuda", batch_size=16, upstream_root=None) -> dict:
    data = read_archive(input_path)
    validate_archive(data, raw=True)
    if Path(output_path).exists() or Path(output_path).with_suffix(".meta.json").exists():
        raise FileExistsError(output_path)
    use_upstream(upstream_root)
    from embodiedskills.language import LanguageEncoder
    from embodiedskills.vision import FrozenPublicVisualEncoder

    visual_path = resolve_offline_model(visual_model, visual_revision)
    language_path = resolve_offline_model(language_model, language_revision)
    visual = FrozenPublicVisualEncoder(backbone="vjepa2", model=visual_path, device=device, spatial_grid=1)
    language = LanguageEncoder(language_path, device=device)
    language_cache = build_language_cache(data["instruction"].astype(str).tolist(), language)
    output = encode_arrays(data, visual, language, batch_size=batch_size, language_cache=language_cache)
    metadata = {
        "schema": "g1_embodiedskills.features.v1", "source_sha256": _digest(input_path),
        "dimensions": validate_archive(output),
        "encoders": {
            "visual_backbone": "vjepa2", "visual_model": str(visual_model),
            "visual_revision": encoder_revision(visual_model, visual_revision),
            "language_model": str(language_model),
            "language_revision": encoder_revision(language_model, language_revision),
            "spatial_grid": 1,
        },
        "visual_preprocessing": visual.preprocessing_contract,
        "visual_normalization": "L2 after spatial_grid=1 pooling",
        "language_preprocessing": "MiniLM masked mean pooling and L2; max_length=64; single instruction batch=1; exact cached float32",
        "g1_language_cache": language_cache,
        "labels": "Copied unchanged from raw archive; measured return/expert utility required",
    }
    _save_archive(output_path, output, metadata)
    return metadata


def episode_split(data: dict, *, validation_fraction=0.2, test_fraction=0.2, seed=0, raw=False) -> dict[str, np.ndarray]:
    validate_archive(data, raw=raw)
    if not 0 < validation_fraction < 1 or not 0 <= test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Require validation > 0, test >= 0 and validation+test < 1")
    rng = np.random.default_rng(seed)
    parts = {"train": [], "validation": [], "test": []}
    for task in np.unique(data["task"]):
        episodes = np.unique(data["seed"][data["task"] == task])
        rng.shuffle(episodes)
        n_val = max(1, int(round(len(episodes) * validation_fraction)))
        n_test = max(1, int(round(len(episodes) * test_fraction))) if test_fraction else 0
        if n_val + n_test >= len(episodes):
            raise ValueError(f"Task {task} needs more episodes for nonempty disjoint splits")
        boundaries = {"validation": episodes[:n_val], "test": episodes[n_val:n_val+n_test], "train": episodes[n_val+n_test:]}
        for name, seeds in boundaries.items():
            parts[name].extend(np.flatnonzero((data["task"] == task) & np.isin(data["seed"], seeds)).tolist())
    return {key: np.asarray(sorted(value), dtype=np.int64) for key, value in parts.items() if value}


def split_archive(input_path, output_dir, *, validation_fraction=0.2, test_fraction=0.2, seed=0, raw=False) -> dict:
    data = read_archive(input_path)
    indices = episode_split(data, validation_fraction=validation_fraction, test_fraction=test_fraction, seed=seed, raw=raw)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    # Keep only row fields understood by the upstream loader; text provenance is a sidecar.
    keys = (*IDENTITIES, *NUMERIC, *(("rgb", "future_rgb", "instruction") if raw else FEATURES))
    inherited = Path(input_path).with_suffix(".meta.json")
    source_metadata = json.loads(inherited.read_text()) if inherited.is_file() else {}
    report = {"schema": "g1_embodiedskills.episode_split.v1", "seed": seed, "source_sha256": _digest(input_path), "splits": {}}
    output_dir.mkdir(parents=True)
    for name, rows in indices.items():
        subset = {key: data[key][rows] for key in keys}
        summary = validate_archive(subset, raw=raw)
        identities = sorted(set(zip(subset["task"].astype(str).tolist(), subset["seed"].tolist())))
        metadata = {**source_metadata, "split": name, "split_seed": seed, "source_sha256": report["source_sha256"], "dimensions": summary, "episodes": identities}
        _save_archive(output_dir / f"{name}.npz", subset, metadata)
        report["splits"][name] = summary
    _write_json(output_dir / "split.json", report)
    return report


def training_config(train: dict, validation: dict, config: dict | None = None) -> dict:
    summaries = [validate_archive(data) for data in (train, validation)]
    resolved = copy.deepcopy(DEFAULT_CONFIG if config is None else config)
    if resolved.get("family") != "continuous":
        raise ValueError("The G1 adapter supports the continuous EmbodiedSkills family")
    model = resolved.setdefault("model", {})
    for key in ("state_dim", "action_dim", "execution_horizon"):
        actual = summaries[0][key]
        if summaries[1][key] != actual:
            raise ValueError(f"Train/validation {key} differs")
        if key in model and model[key] not in (None, "auto", actual):
            raise ValueError(f"Configured {key}={model[key]} differs from archive {actual}")
        model[key] = actual
    train_episodes = set(zip(train["task"].astype(str), train["seed"].tolist()))
    val_episodes = set(zip(validation["task"].astype(str), validation["seed"].tolist()))
    if train_episodes & val_episodes:
        raise ValueError("Train/validation episode overlap at (task,seed)")
    if set(validation["task"].astype(str)) - set(train["task"].astype(str)):
        raise ValueError("Validation tasks require training return statistics for the same tasks")
    if summaries[0]["rows"] * 12 <= 32:
        raise ValueError("Original consensus PCA requires at least three training anchors")
    return resolved


def train_selector(train_path, validation_path, output, *, config=None, device="cuda", upstream_root=None) -> dict:
    """Train the original three members/head and verify its standard runtime bundle."""
    train, validation = read_archive(train_path), read_archive(validation_path)
    resolved = training_config(train, validation, config)
    package_path = use_upstream(upstream_root)
    from embodiedskills.runtime import Selector
    from embodiedskills.training import train as upstream_train

    metadata = []
    for path in (train_path, validation_path):
        sidecar = Path(path).with_suffix(".meta.json")
        metadata.append(json.loads(sidecar.read_text()) if sidecar.is_file() else {})
    encoders = [item.get("encoders") for item in metadata]
    language_cache = training_language_cache(train, validation, metadata)
    if all(encoders):
        if encoders[0] != encoders[1]:
            raise ValueError("Training and validation encoder contracts differ")
        if config is not None and resolved.get("encoders", encoders[0]) != encoders[0]:
            raise ValueError("Configured encoders differ from recorded feature extraction")
        resolved["encoders"] = encoders[0]
    upstream_train(resolved, train_path, validation_path, output, device=device)
    # Keep the standard schema/members/head/statistics untouched. The upstream
    # loader ignores additional metadata fields and still loads weights-only.
    attach_language_cache(Path(output) / "selector.pt", language_cache)
    selector = Selector(Path(output) / "selector.pt", device="cpu")
    first = {key: validation[key][:1] for key in ("visual", "state", "language", "progress", "actions", "mask")}
    prediction = selector.predict(**first)
    if prediction["scores"].shape != (1, 5) or prediction["selected"].shape != (1,) or not np.isfinite(prediction["scores"]).all():
        raise ValueError("Trained bundle failed upstream Selector.predict sanity check")
    revision = subprocess.run(["git", "-C", str(package_path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = {
        "schema": "g1_embodiedskills.training.v1", "upstream_revision": revision.stdout.strip() or None,
        "train_sha256": _digest(train_path), "validation_sha256": _digest(validation_path),
        "train": validate_archive(train), "validation": validate_archive(validation),
        "configuration": resolved, "bundle_load": "passed", "members": len(selector.members),
        "algorithm": "Original EmbodiedSkills continuous training/losses/validation are unchanged",
        "g1_language_cache": {"schema": language_cache["schema"], "encoding": language_cache["encoding"], "sha256": language_cache["sha256"]},
        "member_validation": "Original cosine_gain + 0.01 * expert_correct; not task-success checkpoint selection",
        "evaluation_scope": "Bundle compatibility only; closed-loop task success requires simulator evaluation",
    }
    _write_json(Path(output) / "training_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="Read-only dependency and local encoder-cache inventory")
    inventory.add_argument("--visual-model", default=DEFAULT_VISUAL)
    inventory.add_argument("--language-model", default=DEFAULT_LANGUAGE)
    validate = commands.add_parser("validate", help="Validate actual labels, shapes and identities")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--raw", action="store_true")
    extract = commands.add_parser("extract", help="Raw NPZ to frozen V-JEPA2/MiniLM feature NPZ; offline only")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--visual-model", default=DEFAULT_VISUAL)
    extract.add_argument("--language-model", default=DEFAULT_LANGUAGE)
    extract.add_argument("--visual-revision", help="Cached model SHA; defaults to the pinned upstream model")
    extract.add_argument("--language-revision", help="Cached model SHA; defaults to the pinned upstream model")
    extract.add_argument("--batch-size", type=int, default=16)
    split = commands.add_parser("split", help="Split whole (task,seed) episodes without leakage")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--validation-fraction", type=float, default=0.2)
    split.add_argument("--test-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--raw", action="store_true")
    train = commands.add_parser("train", help="Original three-WAM training and standard bundle verification")
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--config", type=Path, help="Upstream YAML; omit S/A/H or set them to auto")
    for item in (extract, train):
        item.add_argument("--device", default="cuda")
        item.add_argument("--embodiedskills-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        report = encoder_inventory(args.visual_model, args.language_model)
    elif args.command == "validate":
        report = validate_archive(read_archive(args.input), raw=args.raw)
    elif args.command == "extract":
        report = extract_features(args.input, args.output, visual_model=args.visual_model, language_model=args.language_model, visual_revision=args.visual_revision, language_revision=args.language_revision, batch_size=args.batch_size, device=args.device, upstream_root=args.embodiedskills_root)
    elif args.command == "split":
        report = split_archive(args.input, args.output_dir, validation_fraction=args.validation_fraction, test_fraction=args.test_fraction, seed=args.seed, raw=args.raw)
    else:
        config = None
        if args.config:
            import yaml
            config = yaml.safe_load(args.config.read_text())
        report = train_selector(args.train, args.validation, args.output, config=config, device=args.device, upstream_root=args.embodiedskills_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
