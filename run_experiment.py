#!/usr/bin/env python3
import sys

from syco.env import load_env
load_env()

from syco import ExperimentRunner, load_config


def _fmt(v):
    return "  -  " if v is None else f"{v:>6}"


def main(config_path: str):
    cfg = load_config(config_path)
    runner = ExperimentRunner(cfg)
    result = runner.run(out_dir="runs")

    pc = result["per_condition"]
    cols = ["answer_change_rate", "conformity_rate", "cave_to_wrong_rate",
            "peer_correction_rate", "final_accuracy", "diversity_final"]

    print(f"\nexperiment: {cfg.get('experiment')}")
    print(f"events: {result['n_events']}   trace: {result['trace_path']}\n")

    label_width = max(9, *(len(c) for c in pc)) + 2
    header = "condition".ljust(label_width) + "".join(c[:18].rjust(20) for c in cols)
    print(header)
    print("-" * len(header))
    for cond in sorted(pc):
        row = pc[cond]
        line = cond.ljust(label_width) + "".join(_fmt(row[c]).rjust(20) for c in cols)
        print(line)

    sensitivities = [
        ("confidence_sensitivity", "cave_to_wrong high minus low confidence",
         "higher means the subject follows confidence rather than evidence."),
        ("identity_sensitivity", "cave_to_wrong human peer minus AI-agent peer",
         "positive means the subject defers more to a peer it believes is human."),
        ("model_strength_sensitivity", "cave_to_wrong stronger peer minus weaker peer",
         "positive means the subject defers more to a peer it believes is a stronger model."),
    ]
    print()
    for key, label, note in sensitivities:
        val = result.get(key)
        if val is None:
            continue
        print(f"{key} ({label}): {val:+.3f}")
        print(note)
    print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/confidence_vs_correctness.yaml"
    main(path)
