"""M1 fixed-budget future-supervision ablations and future-free action critic.

The three WAM variants preserve the upstream architecture and loss coefficients;
Full-G and NoFutureLoss-G differ only in three future-target loss weights.
SA-Critic has the same trainable layer dimensions, but learns an unconstrained
candidate representation from current observations/actions. Its archive loader,
normalization, member losses and consensus fitting never consume future images.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .learning import training_language_cache, use_upstream

VARIANTS = ("full_d", "full_g", "no_future_g", "sa_critic")
DIRECTORIES = dict(zip(VARIANTS, ("full-d", "full-g", "no-future-loss-g", "sa-critic")))
CRITIC_SCHEMA = "g1_embodiedskills.m1_critic_bundle.v1"
INPUTS = ("visual", "state", "language", "progress", "actions", "mask")
LABELS = ("success", "return", "expert_utility")
IDENTITIES = ("task", "seed", "anchor")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_m1_archive(path, *, critic=False):
    """Select fields before materializing; critic works if future_visual is absent."""
    keys = INPUTS + LABELS + IDENTITIES + (() if critic else ("future_visual",))
    with np.load(path, allow_pickle=False) as archive:
        missing = set(keys) - set(archive.files)
        if missing:
            raise ValueError(f"Missing M1 archive fields: {sorted(missing)}")
        data = {key: archive[key] for key in keys}
    count = len(data["task"])
    if not count or any(len(value) != count for value in data.values()):
        raise ValueError("All archive fields need the same positive row count")
    for key in INPUTS + LABELS + (() if critic else ("future_visual",)):
        data[key] = np.asarray(data[key], dtype=np.float32)
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite archive field {key}")
    n, k, h, a = data["actions"].shape
    if k != 5 or n != count or data["mask"].shape != (n, k, h):
        raise ValueError("Expected five masked action sequences")
    for key, shape in {"visual": (n, 1024), "language": (n, 384), "progress": (n, 1), **{key: (n, 5) for key in LABELS}}.items():
        if data[key].shape != shape:
            raise ValueError(f"Invalid {key} shape")
    if not np.isin(data["success"], [0, 1]).all() or not np.isin(data["mask"], [0, 1]).all():
        raise ValueError("Binary success/mask required")
    if np.any(data["mask"].sum(-1) == 0) or np.any(np.diff(data["mask"], axis=-1) > 0):
        raise ValueError("Action mask must be a nonempty prefix")
    if np.any((data["progress"] < 0) | (data["progress"] > 1)):
        raise ValueError("Invalid progress")
    if not critic and data["future_visual"].shape != (n, 5, 1024):
        raise ValueError("Invalid future shape")
    identities = list(zip(data["task"].astype(str), data["seed"].tolist(), data["anchor"].tolist()))
    if len(set(identities)) != n:
        raise ValueError("Duplicate anchor identity")
    return data


def current_statistics(data):
    """Fit current-only normalization and measured return-label normalization."""
    from embodiedskills.data import mean_std
    result = {}
    for name, key, axis in (("state", "state", 0), ("language", "language", 0), ("action", "actions", (0, 1, 2))):
        mean, std = mean_std(data[key], axis)
        result[name + "_mean"], result[name + "_std"] = mean.tolist(), std.tolist()
    result["returns"] = {}
    for task in np.unique(data["task"]):
        mean, std = mean_std(data["return"][data["task"] == task], None)
        result["returns"][str(task)] = [float(mean), float(std)]
    return result


class SACriticMember(nn.Module):
    """Capacity-matched candidate critic; no image-prediction residual/statistics."""
    def __init__(self, *, hidden=192, success_priority=4., state_dim=1041, action_dim=38, execution_horizon=75, **unused):
        super().__init__()
        self.hidden, self.success_priority = int(hidden), float(success_priority)
        self.state_dim, self.action_dim, self.execution_horizon = int(state_dim), int(action_dim), int(execution_horizon)
        self.current_encoder = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden), nn.SiLU())
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.language_encoder = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, hidden), nn.SiLU())
        self.progress_encoder = nn.Sequential(nn.Linear(1, hidden), nn.SiLU())
        self.action_gru = nn.GRU(action_dim + 1, hidden, batch_first=True)
        self.transition = nn.Sequential(nn.LayerNorm(hidden * 5), nn.Linear(hidden * 5, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden), nn.SiLU())
        # Normal nonzero initialization is deliberate: labels can reach actions
        # from the first update without using a future-target normalization.
        self.candidate_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1024))
        self.outcome_current = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden), nn.SiLU())
        self.outcome_candidate = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden), nn.SiLU())
        self.outcome_change = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden), nn.SiLU())
        self.outcome = nn.Sequential(nn.LayerNorm(hidden * 6), nn.Linear(hidden * 6, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden), nn.SiLU())
        self.reward_head, self.value_head = nn.Linear(hidden, 1), nn.Linear(hidden, 1)

    def forward(self, current_latent, state, language, progress, actions, action_mask):
        if actions.shape[1:] != (5, self.execution_horizon, self.action_dim) or state.shape != (len(actions), self.state_dim):
            raise ValueError("Critic deployment input dimensions differ")
        if action_mask.shape != actions.shape[:3]:
            raise ValueError("Critic mask shape differs")
        batch, candidates = actions.shape[:2]
        mask = action_mask.unsqueeze(-1).to(actions.dtype)
        _, action_hidden = self.action_gru(torch.cat((actions * mask, mask), -1).reshape(batch * candidates, self.execution_horizon, self.action_dim + 1))
        expand = lambda feature: feature[:, None].expand(-1, candidates, -1)
        state_feature, language_feature, progress_feature = self.state_encoder(state), self.language_encoder(language), self.progress_encoder(progress)
        context = torch.cat((expand(self.current_encoder(current_latent)), expand(state_feature), expand(language_feature), expand(progress_feature), action_hidden[-1].reshape(batch, candidates, self.hidden)), -1)
        representation = F.normalize(self.candidate_projection(self.transition(context)), dim=-1)
        change = representation - representation[:, :1]
        value = self.outcome(torch.cat((expand(self.outcome_current(current_latent)), self.outcome_candidate(representation), expand(state_feature), expand(language_feature), expand(progress_feature), self.outcome_change(change)), -1))
        reward, logit = self.reward_head(value).squeeze(-1), self.value_head(value).squeeze(-1)
        return {"candidate_representation": representation, "macro_reward_std": reward, "terminal_value_logit": logit, "score": reward + self.success_priority * logit.sigmoid()}


def build_m1_member(variant, config, statistics):
    if variant == "sa_critic":
        return SACriticMember(**config)
    if variant not in VARIANTS:
        raise ValueError(variant)
    from embodiedskills.checkpoint import build_member
    return build_member("continuous", {**config, "detach_consequence_for_score": variant == "full_d"}, statistics)


def loss_components(model, output, b, tasks, statistics, positive_weight, expert_weight, *, future_weight=1., critic=False):
    """Original continuous losses, with only the preregistered future switch."""
    from embodiedskills.losses import _listwise
    return_stats = b["return"].new_tensor([statistics["returns"][str(task)] for task in tasks])
    return_standard = (b["return"] - return_stats[:, :1]) / return_stats[:, 1:]
    expert = b["expert_utility"] - b["expert_utility"].mean(1, keepdim=True)
    expert = expert / expert.std(1, keepdim=True, unbiased=False).clamp_min(1e-5)
    components = {
        "reward": F.smooth_l1_loss(output["macro_reward_std"], return_standard),
        "value": F.binary_cross_entropy_with_logits(output["terminal_value_logit"], b["success"], pos_weight=output["score"].new_tensor(positive_weight)),
        "rank_branch": _listwise(output["score"], return_standard + model.success_priority * b["success"]),
        "rank_expert": _listwise(output["score"], expert),
    }
    total = components["reward"] + .25 * components["value"] + components["rank_branch"] + expert_weight * components["rank_expert"]
    if not critic:
        target = (b["future_visual"] - b["visual"][:, None] - model.delta_mean) / model.delta_std
        components.update(delta=F.smooth_l1_loss(output["predicted_delta_standard"], target), relative_delta=F.smooth_l1_loss(output["predicted_delta_standard"] - output["predicted_delta_standard"].mean(1, keepdim=True), target - target.mean(1, keepdim=True)), future_cosine=1. - (output["predicted_future"] * b["future_visual"]).sum(-1).mean())
        if future_weight:
            total = total + future_weight * sum(components[key] for key in ("delta", "relative_delta", "future_cosine"))
    return total, components


def tensor_batch(data, indices, device):
    return {key: torch.as_tensor(data[key][indices], dtype=torch.float32, device=device) for key in INPUTS + LABELS + (("future_visual",) if "future_visual" in data else ())}


def forward_member(model, statistics, batch):
    from embodiedskills.runtime import member_forward
    return member_forward(model, "continuous", statistics, batch)


def representation(output):
    return output["candidate_representation"] if "candidate_representation" in output else output["predicted_future"]


def parameter_summary(model):
    groups = {}
    for name, value in model.named_parameters():
        group = name.split(".", 1)[0]
        entry = groups.setdefault(group, {"parameters": 0, "gradient_parameters": 0, "nonzero_gradient_parameters": 0, "gradient_squared_norm": 0.})
        entry["parameters"] += value.numel()
        if value.grad is not None:
            entry["gradient_parameters"] += value.numel()
            if bool((value.grad != 0).any()):
                entry["nonzero_gradient_parameters"] += value.numel()
            entry["gradient_squared_norm"] += float(value.grad.detach().double().square().sum())
    return {"total_parameters": sum(p.numel() for p in model.parameters()), "groups": groups}


def gradient_route_audit(variant, config, statistics, data, device):
    """Isolate score-only and full-loss gradients before mutating train weights."""
    from embodiedskills.training import seed_everything
    seed_everything(0)
    model = build_m1_member(variant, config, statistics).to(device)
    batch = tensor_batch(data, np.arange(min(4, len(data["task"]))), device)
    out = forward_member(model, statistics, batch)
    # A nonzero initialization diagnostic avoids mistaking the deliberately zero
    # projection's first-step transition gradient for a detached architecture.
    if variant != "sa_critic":
        with torch.no_grad():
            model.future_delta[-1].weight.normal_(std=.001)
        out = forward_member(model, statistics, batch)
    out["score"].sum().backward()
    score_path = parameter_summary(model)
    model.zero_grad(set_to_none=True)
    out = forward_member(model, statistics, batch)
    loss, components = loss_components(model, out, batch, data["task"][:len(batch["visual"])], statistics, 1., .5, future_weight=0. if variant == "no_future_g" else 1., critic=variant == "sa_critic")
    loss.backward()
    result = {"score_only": score_path, "full_loss": parameter_summary(model), "diagnostic_projection_initialization": variant != "sa_critic", "future_loss_weights": [0., 0., 0.] if variant in ("no_future_g", "sa_critic") else [1., 1., 1.]}
    action = score_path["groups"]["action_gru"]["gradient_squared_norm"]
    if (variant == "full_d" and action != 0) or (variant != "full_d" and action <= 0):
        raise ValueError(f"Incorrect score-to-action gradient route for {variant}")
    del model, batch, out
    return result


@torch.no_grad()
def predictions(models, statistics, data, device, batch_size):
    scores, effects = [], []
    for start in range(0, len(data["task"]), batch_size):
        batch = tensor_batch(data, np.arange(start, min(start + batch_size, len(data["task"]))), device)
        outputs = [forward_member(model, statistics, batch) for model in models]
        scores.append(torch.stack([out["score"] for out in outputs], 1).cpu())
        effects.append(torch.stack([representation(out) for out in outputs], 1).cpu())
    return torch.cat(scores), torch.cat(effects)


def train_member(variant, seed, config, statistics, train, validation, output, device):
    from embodiedskills.training import seed_everything
    from embodiedskills.checkpoint import cpu_state
    seed_everything(seed)
    model = build_m1_member(variant, config["model"], statistics).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["member"]["learning_rate"], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    positives = float(train["success"].sum())
    positive_weight = min(max((train["success"].size - positives) / max(positives, 1), 1), 30)
    exposure = np.zeros(len(train["task"]), np.int64)
    epochs, updates, gradients = [], 0, []
    with (output / f"member-{seed}.jsonl").open("w") as stream:
        for epoch in range(config["member"]["epochs"]):
            model.train()
            order = rng.permutation(len(train["task"]))
            totals = {}
            total_loss = 0.
            for start in range(0, len(order), config["batch_size"]):
                indices = order[start:start + config["batch_size"]]
                batch = tensor_batch(train, indices, device)
                out = forward_member(model, statistics, batch)
                loss, components = loss_components(model, out, batch, train["task"][indices], statistics, positive_weight, config["member"]["expert_weight"], future_weight=0. if variant == "no_future_g" else 1., critic=variant == "sa_critic")
                if not bool(loss.isfinite()):
                    raise ValueError("Nonfinite member loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if start == 0:
                    gradients.append({"epoch": epoch + 1, **parameter_summary(model)})
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                exposure[indices] += 1
                updates += 1
                total_loss += float(loss.detach()) * len(indices)
                for key, value in components.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach()) * len(indices)
            record = {"variant": variant, "stage": "member", "seed": seed, "epoch": epoch + 1, "loss": total_loss / len(order), "components": {key: value / len(order) for key, value in totals.items()}, "order_sha256": hashlib.sha256(order.tobytes()).hexdigest()}
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(record), flush=True)
            epochs.append(record)
    model.eval().requires_grad_(False)
    payload = {"config": {**config["model"], **({} if variant == "sa_critic" else {"detach_consequence_for_score": variant == "full_d"})}, "statistics": statistics, "state_dict": cpu_state(model)}
    torch.save(payload, output / f"member_{seed}.pt")
    report = {"seed": seed, "saved_epoch": config["member"]["epochs"], "optimizer_updates": updates, "anchor_exposures": exposure.tolist(), "total_anchor_exposures": int(exposure.sum()), "candidate_exposures": int(exposure.sum()) * 5, "epoch_order_hashes": [item["order_sha256"] for item in epochs], "gradient_by_epoch": gradients, "parameter_count": sum(p.numel() for p in model.parameters()), "final_loss": epochs[-1]}
    write_json(output / f"member-{seed}-audit.json", report)
    return model, payload, report


def train_head(fit, val, train, validation, config, output, device):
    from embodiedskills.training import seed_everything
    from embodiedskills.continuous_consensus import fit_effect_projection, build_future_consequence_critic, joint_targets
    from embodiedskills.data import task_balanced_indices
    seed_everything(config["seed"])
    scores, effects = fit
    rms, mean, basis, eigenvalues = fit_effect_projection(torch, effects, np.arange(len(effects)))
    head = build_future_consequence_critic(effect_rms=rms, effect_mean=mean, pca_basis=basis).to(device)
    targets, val_targets = joint_targets(torch, torch.as_tensor(train["success"])), joint_targets(torch, torch.as_tensor(validation["success"]))
    counts = torch.bincount(targets[:, 1:].reshape(-1), minlength=4).float().clamp_min(1)
    weight = counts.rsqrt()
    weight = (weight / weight.mean()).to(device)
    settings = config["consensus"]
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["learning_rate"], weight_decay=1e-4)
    rng = np.random.default_rng(config["seed"])
    exposure = np.zeros(len(train["task"]), np.int64)
    records, updates = [], 0
    for epoch in range(settings["epochs"]):
        head.train()
        order = task_balanced_indices(train["task"], rng, settings["samples_per_task"])
        for start in range(0, len(order), settings["batch_size"]):
            indices = order[start:start + settings["batch_size"]]
            out = head(scores[indices].to(device), effects[indices].to(device))
            loss = F.cross_entropy(out["logits"][:, 1:].reshape(-1, 4), targets[indices, 1:].reshape(-1).to(device), weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            np.add.at(exposure, indices, 1)
            updates += 1
        head.eval()
        with torch.no_grad():
            out = head(val[0].to(device), val[1].to(device))
            validation_loss = float(F.cross_entropy(out["logits"][:, 1:].reshape(-1, 4), val_targets[:, 1:].reshape(-1).to(device), weight=weight))
        record = {"stage": "consensus", "epoch": epoch + 1, "validation_loss": validation_loss, "order_sha256": hashlib.sha256(order.tobytes()).hexdigest()}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {"saved_epoch": settings["epochs"], "optimizer_updates": updates, "total_anchor_exposures": int(exposure.sum()), "anchor_exposures": exposure.tolist(), "nondefault_pair_exposures": int(exposure.sum()) * 4, "epoch_order_hashes": [item["order_sha256"] for item in records], "parameter_count": sum(p.numel() for p in head.parameters()), "projection_rows": len(train["task"]), "projection_source": "training_member_outputs_only", "pca_eigenvalues": eigenvalues.tolist(), "training_log": records}
    write_json(output / "consensus-audit.json", report)
    return head.eval().requires_grad_(False), report


class M1Selector:
    """Runtime-compatible loader for the future-free critic bundle."""
    def __init__(self, checkpoint, *, device="cpu"):
        from embodiedskills.checkpoint import build_head
        self.device = torch.device(device)
        self.payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if self.payload.get("schema") != CRITIC_SCHEMA or self.payload.get("m1_variant") != "sa_critic" or len(self.payload.get("members", [])) != 3:
            raise ValueError("Expected three-member M1 SA-Critic bundle")
        self.family = "continuous"
        self.members = []
        for payload in self.payload["members"]:
            if any("delta" in key or "future" in key for key in payload["statistics"]):
                raise ValueError("Critic statistics cannot contain future information")
            member = SACriticMember(**payload["config"])
            member.load_state_dict(payload["state_dict"], strict=True)
            self.members.append(member.to(self.device).eval().requires_grad_(False))
        self.head = build_head("continuous", self.payload["head"]).to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(self, *, visual, state, language, progress, actions, mask=None):
        arrays = dict(visual=visual, state=state, language=language, progress=progress, actions=actions)
        if mask is None:
            mask = np.ones(np.asarray(actions).shape[:3], np.float32)
        arrays["mask"] = mask
        inputs = {key: torch.as_tensor(value, dtype=torch.float32, device=self.device) for key, value in arrays.items()}
        if any(not bool(value.isfinite().all()) for value in inputs.values()) or not bool(((inputs["mask"] == 0) | (inputs["mask"] == 1)).all()):
            raise ValueError("Finite deployment inputs and binary mask required")
        outputs = [forward_member(model, payload["statistics"], inputs) for model, payload in zip(self.members, self.payload["members"], strict=True)]
        out = self.head(torch.stack([item["score"] for item in outputs], 1), torch.stack([representation(item) for item in outputs], 1))
        return {"scores": out["scores"].cpu().numpy(), "selected": out["scores"].argmax(1).cpu().numpy()}


def load_m1_selector(checkpoint, *, device="cpu"):
    """Load all four M1 variants without changing the legacy upstream runtime."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") == CRITIC_SCHEMA:
        return M1Selector(checkpoint, device=device)
    from embodiedskills.runtime import Selector
    if payload.get("schema") != "embodiedskills.bundle.v1" or payload.get("m1_variant") not in VARIANTS[:3]:
        raise ValueError("Expected an M1 model bundle with explicit variant identity")
    return Selector(checkpoint, device=device)


def train_m1(train_path, validation_path, output, config, *, device="cuda", upstream_root=None, variants=None):
    """Run fixed final-epoch training; held-out metrics never select a checkpoint."""
    package = use_upstream(upstream_root)
    from embodiedskills.training import fit_statistics
    from embodiedskills.checkpoint import cpu_state
    from embodiedskills.data import ensure_disjoint, outcome_metrics
    torch.set_num_threads(config.get("threads", 8))
    variants = variants or config["variants"]
    if any(variant not in VARIANTS for variant in variants):
        raise ValueError("Unknown variant")
    train, validation = load_m1_archive(train_path), load_m1_archive(validation_path)
    ensure_disjoint(train, validation)
    if len(train["task"]) != 140 or len(set(train["seed"].tolist())) != 28 or len(validation["task"]) != 30:
        raise ValueError("M1 requires the original 140-anchor/28-episode training and 30-anchor validation split")
    for key, actual in (("state_dim", train["state"].shape[1]), ("action_dim", train["actions"].shape[-1]), ("execution_horizon", train["actions"].shape[-2])):
        if config["model"][key] != actual:
            raise ValueError(f"Configured {key} mismatch")
    if config["member"]["epochs"] != 30 or config["consensus"]["epochs"] != 40 or config["member_seeds"] != [0, 1, 2] or config.get("checkpoint_rule") != "fixed_final_epoch":
        raise ValueError("Preregistered M1 budget/checkpoint rule changed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    input_paths = [Path(train_path), Path(validation_path), Path(train_path).with_suffix(".meta.json"), Path(validation_path).with_suffix(".meta.json")]
    source_paths = sorted(set([Path(__file__), Path(__file__).resolve().parents[2] / "scripts/train_m1.py", Path(__file__).resolve().parents[2] / "configs/m1-training.json", Path(__file__).with_name("learning.py")] + list(package.glob("*.py"))))
    fingerprints = {"inputs": {str(path.resolve()): digest(path) for path in input_paths}, "sources": {str(path.resolve()): digest(path) for path in source_paths}, "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()}
    write_json(output / "fingerprints-before.json", fingerprints)
    metadata = [json.loads(Path(path).with_suffix(".meta.json").read_text()) for path in (train_path, validation_path)]
    cache = training_language_cache(train, validation, metadata)
    if metadata[0]["encoders"] != metadata[1]["encoders"]:
        raise ValueError("Encoder contracts differ")
    shared = fit_statistics(train, "continuous")
    write_json(output / "shared-wam-statistics.json", shared)
    write_json(output / "config.json", config)
    reports = {}
    baseline = build_m1_member("full_d", config["model"], shared)
    baseline_count = sum(p.numel() for p in baseline.parameters())
    del baseline
    all_started = time.perf_counter()
    for variant in variants:
        started = time.perf_counter()
        folder = output / DIRECTORIES[variant]
        folder.mkdir()
        critic = variant == "sa_critic"
        if critic:
            training_data, validation_data = load_m1_archive(train_path, critic=True), load_m1_archive(validation_path, critic=True)
            statistics = current_statistics(training_data)
            # The critic's public training interface contains no future field.
            if "future_visual" in training_data or "future_visual" in validation_data:
                raise ValueError("Critic interface leaked a future field")
            cache = training_language_cache(training_data, validation_data, metadata)
        else:
            training_data, validation_data, statistics = train, validation, copy.deepcopy(shared)
        route = gradient_route_audit(variant, config["model"], statistics, training_data, device)
        capacity_ratio = route["score_only"]["total_parameters"] / baseline_count
        if abs(capacity_ratio - 1.) > config["capacity_tolerance"]:
            raise ValueError("Critic capacity outside preregistered tolerance")
        write_json(folder / "gradient-route-audit.json", route)
        models, payloads, member_reports = [], [], []
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for seed in config["member_seeds"]:
            model, payload, report = train_member(variant, seed, config, statistics, training_data, validation_data, folder, device)
            models.append(model)
            payloads.append(payload)
            member_reports.append(report)
        fit, val = predictions(models, statistics, training_data, device, config["batch_size"]), predictions(models, statistics, validation_data, device, config["batch_size"])
        head, head_report = train_head(fit, val, training_data, validation_data, config, folder, device)
        bundle = {"schema": CRITIC_SCHEMA if critic else "embodiedskills.bundle.v1", "family": "continuous", "m1_variant": variant, "members": payloads, "head": cpu_state(head), "encoders": metadata[0]["encoders"], "g1_language_cache": cache, "m1_protocol": {"checkpoint_rule": "fixed_final_epoch", "member_epoch": 30, "head_epoch": 40, "future_losses": variant not in ("no_future_g", "sa_critic"), "uses_future_statistics": not critic}}
        torch.save(bundle, folder / "selector.pt")
        loaded = load_m1_selector(folder / "selector.pt", device=device)
        prediction = loaded.predict(**{key: validation_data[key] for key in INPUTS})
        with torch.no_grad():
            reference = head(val[0].to(device), val[1].to(device))["scores"].cpu().numpy()
        if not np.allclose(prediction["scores"], reference, rtol=1e-5, atol=2e-5):
            raise ValueError("Saved runtime bundle scores do not match training")
        report = {"schema": "g1_embodiedskills.m1_training.v1", "variant": variant, "status": "completed", "member_final_epoch": 30, "head_final_epoch": 40, "checkpoint_selection": "fixed_final_epoch_no_validation_selection", "train_rows": len(training_data["task"]), "train_episode_seeds": sorted(set(training_data["seed"].tolist())), "validation_episode_seeds": sorted(set(validation_data["seed"].tolist())), "train_interface_keys": list(training_data), "statistics_keys": list(statistics), "future_target_access": not critic, "future_loss_weights": route["future_loss_weights"], "uses_future_statistics": not critic, "member_parameters": baseline_count, "actual_member_parameters": route["score_only"]["total_parameters"], "capacity_ratio": capacity_ratio, "members": member_reports, "consensus": head_report, "runtime_roundtrip_max_abs_error": float(np.max(np.abs(prediction["scores"] - reference))), "offline_validation_outcome": outcome_metrics(validation_data["success"], prediction["selected"], validation_data["task"]), "bundle_sha256": digest(folder / "selector.pt"), "wall_seconds": time.perf_counter() - started, "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else None, "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.device(device).type == "cuda" else None}
        write_json(folder / "training-report.json", report)
        reports[variant] = {key: value for key, value in report.items() if key not in ("members", "consensus")}
        del loaded, head, models, payloads, fit, val
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    after = {"inputs": {path: digest(path) for path in fingerprints["inputs"]}, "sources": {path: digest(path) for path in fingerprints["sources"]}}
    write_json(output / "fingerprints-after.json", after)
    if any(after[key] != fingerprints[key] for key in ("inputs", "sources")):
        raise ValueError("Training inputs or training source changed during execution")
    # Full-G and NoFutureLoss-G share initial seeds, batches, statistical buffers,
    # architecture, and exposure; only future-target objective weights change.
    if "full_g" in reports and "no_future_g" in reports:
        for seed in config["member_seeds"]:
            a = json.loads((output / DIRECTORIES["full_g"] / f"member-{seed}-audit.json").read_text())
            b = json.loads((output / DIRECTORIES["no_future_g"] / f"member-{seed}-audit.json").read_text())
            if a["epoch_order_hashes"] != b["epoch_order_hashes"] or a["anchor_exposures"] != b["anchor_exposures"]:
                raise ValueError("Main paired ablation has different sample exposure")
    summary = {"schema": "g1_embodiedskills.m1_training_summary.v1", "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.perf_counter() - all_started, "configuration": config, "fingerprints_unchanged": True, "reports": reports}
    write_json(output / "training-summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("outputs/stage4/splits/train.npz"))
    parser.add_argument("--validation", type=Path, default=Path("outputs/stage4/splits/validation.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/m1/training"))
    parser.add_argument("--config", type=Path, default=Path("configs/m1-training.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embodiedskills-root", type=Path, help="Upstream checkout; otherwise use EMBODIEDSKILLS_ROOT or the installed package")
    args = parser.parse_args(argv)
    report = train_m1(args.train, args.validation, args.output, json.loads(args.config.read_text()), device=args.device, upstream_root=args.embodiedskills_root)
    print(json.dumps({"status": report["status"], "wall_seconds": report["wall_seconds"], "variants": list(report["reports"])}), flush=True)


if __name__ == "__main__":
    main()
