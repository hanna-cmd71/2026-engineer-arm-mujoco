"""Plot recorded JGarm target, position, and torque curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="label=csv_path; repeat for PD and impedance.",
    )
    parser.add_argument("--joint", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/control_verification"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.joint <= 6:
        raise SystemExit("--joint must be between 1 and 6")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for item in args.input:
        label, path = item.split("=", 1)
        data = np.genfromtxt(path, delimiter=",", names=True)
        datasets.append((label, data))
    joint = args.joint
    target_name = f"target{joint}"
    position_name = f"q{joint}"
    command_name = f"tau_cmd{joint}"

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    for label, data in datasets:
        axes[0].plot(
            data["time_s"],
            data[target_name],
            "--",
            label=f"{label} target",
        )
        axes[0].plot(
            data["time_s"],
            data[position_name],
            label=f"{label} actual",
        )
        axes[1].plot(
            data["time_s"],
            data[command_name],
            label=f"{label} command",
        )
    axes[0].set_ylabel("position / rad")
    axes[1].set_ylabel("torque / N*m")
    axes[1].set_xlabel("time / s")
    axes[0].legend()
    axes[1].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    figure.suptitle(f"J{joint} target, actual, and command torque")
    position_path = args.output_dir / f"position_and_torque_J{joint}.png"
    figure.tight_layout()
    figure.savefig(position_path, dpi=160)
    plt.close(figure)

    for label, data in datasets:
        figure, axis = plt.subplots(figsize=(10, 4.5))
        for column, name in (
            (command_name, "command"),
            (f"pd{joint}", "PD / impedance"),
            (f"gravity{joint}", "gravity"),
            (f"friction{joint}", "friction"),
            (f"breakaway{joint}", "breakaway"),
            (f"mujoco_friction{joint}", "MuJoCo friction"),
        ):
            axis.plot(data["time_s"], data[column], label=name)
        axis.set_xlabel("time / s")
        axis.set_ylabel("torque / N*m")
        axis.set_title(f"{label}: J{joint} torque components")
        axis.grid(True, alpha=0.3)
        axis.legend()
        output = args.output_dir / f"torque_components_{label}_J{joint}.png"
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plt.close(figure)

    metrics = {}
    for label, data in datasets:
        target = data[target_name]
        position = data[position_name]
        step_index = np.flatnonzero(np.abs(np.diff(target)) > 1e-8)
        step_time = (
            float(data["time_s"][step_index[0] + 1])
            if step_index.size
            else float(data["time_s"][0])
        )
        hold = data["time_s"] < step_time - 0.1
        after = data["time_s"] >= step_time
        error = position - target
        metrics[label] = {
            "step_time_s": step_time,
            "hold_max_error_rad": float(
                np.max(np.abs(error[hold]))
            ),
            "hold_mean_gravity_nm": float(
                np.nanmean(data[f"gravity{joint}"][hold])
            ),
            "step_peak_error_rad": float(
                np.max(np.abs(error[after]))
            ),
            "final_error_rad": float(error[-1]),
            "peak_command_nm": float(
                np.max(np.abs(data[command_name][after]))
            ),
        }
    report = args.output_dir / f"control_metrics_J{joint}.json"
    report.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"plots: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
