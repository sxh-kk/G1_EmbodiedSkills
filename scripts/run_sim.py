"""Run the upstream evaluator through local factory and callback hooks."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g1_embodiedskills.io import write_json, RunFinished


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sonic", type=Path, default=os.environ.get("SONIC_ROOT"))
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["interface", "direct", "collect", "closed_loop"], default="interface")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seconds", type=float, default=8.)
    p.add_argument("--cameras", action="store_true")
    p.add_argument("--task", type=Path)
    p.add_argument("--servo", action="store_true", help="Use measured TCP feedback to correct SONIC references")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--position-jitter", type=float, default=0., help="Cube XY randomization radius in meters")
    p.add_argument("--record-video", action="store_true", help="Record full 5Hz video; otherwise render only sensing/diagnostic frames")
    p.add_argument("--collection-config", type=Path)
    p.add_argument("--hard-reset", action="store_true", help="Rebuild PhysX before each diagnostic repeat")
    p.add_argument("--method", choices=["direct", "random", "selector", "fixed2", "fixed", "phase"], default="selector")
    p.add_argument("--selector", type=Path)
    p.add_argument("--selector-kind", choices=["legacy", "full_d", "full_g", "no_future_g", "sa_critic"], default="legacy")
    p.add_argument("--fixed-candidate-index", type=int, choices=range(5))
    p.add_argument("--phase-policy", type=Path)
    p.add_argument("--amplitude-m", type=float, default=.012)
    p.add_argument("--instruction", default="Place the red block into the green tray.")
    p.add_argument("--embodiedskills-root", dest="upstream_root", type=Path, default=os.environ.get("EMBODIEDSKILLS_ROOT"))
    a = p.parse_args()
    if a.sonic is None:
        p.error("Set SONIC_ROOT or pass --sonic")
    for key in ["sonic", "reference", "asset", "output", "task", "collection_config", "selector", "upstream_root", "phase_policy"]:
        value = getattr(a, key)
        if value is not None:
            setattr(a, key, value.resolve())
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / "request.json", vars(a))
    os.environ["WANDB_MODE"] = "offline"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    sys.path.insert(0, str(a.sonic))
    os.chdir(a.sonic)
    try:
        from gear_sonic import train_agent_trl
        def factory(config, device, args_cli):
            from g1_embodiedskills.simulation import create_env
            return create_env(config, device, args_cli, a)
        train_agent_trl.create_manager_env = factory
        from gear_sonic.eval_agent_trl import main as sonic_main
        os.environ["G1_RUN_REQUEST"] = str(a.output / "request.json")
        sys.argv = [sys.argv[0], "--config-path=" + str(a.sonic / "gear_sonic/config"),
            "+checkpoint=sonic_release/last.pt", "+headless=True", "++use_encoder=g1",
            "++run_eval_loop=False", f"++num_envs={a.num_envs}", "++eval_callbacks=[g1_migration]",
            "+callbacks.g1_migration._target_=g1_embodiedskills.simulation.SimulationCallback",
            "++manager_env.commands.motion.motion_lib_cfg.motion_file=" + str(a.reference / "motion.pkl"),
            "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy",
            "+manager_env/terminations=tracking/eval",
            "++manager_env.config.enable_cameras=" + str(a.cameras),
            "hydra.run.dir=" + str(a.output / "hydra")]
        sonic_main()
        raise RuntimeError("Evaluator exited before writing result")
    except RunFinished as finished:
        result = json.loads((a.output / "result.json").read_text(encoding="utf-8"))
        write_json(a.output / "process_result.json", dict(returncode=finished.code, result_present=bool(result)))
        return finished.code
    except BaseException as error:
        write_json(a.output / "ERROR.json", dict(type=type(error).__name__, error=str(error)))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
