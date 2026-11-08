"""Estimate gain and phase from recorded stepped-sine sweeps."""

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
        default=Path("results/control_verification_sweep"),
    )
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--dwell", type=float, default=None)
    parser.add_argument("--frequencies", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joint = args.joint
    results = {}
    for item in args.input:
        label, path = item.split("=", 1)
        csv_path = Path(path)
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        metadata_path = csv_path.with_suffix(".json")
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        start_time = (
            args.start_time
            if args.start_time is not None
            else float(metadata.get("step_time_s", 1.0))
        )
        dwell = (
            args.dwell
            if args.dwell is not None
            else float(metadata.get("dwell_s", 3.0))
        )
        frequencies = (
            [float(value) for value in args.frequencies.split(",")]
            if args.frequencies is not None
            else [float(value) for value in metadata["frequencies_hz"]]
        )
        target = data[f"target{joint}"]
        actual = data[f"q{joint}"]
        time = data["time_s"]
        gains = []
        phases = []
        for index, frequency in enumerate(frequencies):
            segment_start = start_time + index * dwell
            segment_end = segment_start + dwell
            margin = 0.2 * dwell
            active = (time >= segment_start + margin) & (
                time <= segment_end - 0.05
            )
            if np.count_nonzero(active) < 10:
                gains.append(np.nan)
                phases.append(np.nan)
                continue
            local_time = time[active] - segment_start
            omega = 2.0 * np.pi * frequency
            design = np.column_stack(
                [
                    np.sin(omega * local_time),
                    np.cos(omega * local_time),
                    np.ones(local_time.size),
                ]
            )
            target_coeff, *_ = np.linalg.lstsq(
                design,
                target[active],
                rcond=None,
            )
            actual_coeff, *_ = np.linalg.lstsq(
                design,
                actual[active],
                rcond=None,
            )
            target_amplitude = float(
                np.hypot(target_coeff[0], target_coeff[1])
            )
            actual_amplitude = float(
                np.hypot(actual_coeff[0], actual_coeff[1])
            )
            if target_amplitude < 1e-9:
                gains.append(np.nan)
                phases.append(np.nan)
                continue
            target_phase = float(
                np.arctan2(target_coeff[1], target_coeff[0])
            )
            actual_phase = float(
                np.arctan2(actual_coeff[1], actual_coeff[0])
            )
            gains.append(actual_amplitude / target_amplitude)
            phases.append(actual_phase - target_phase)
        results[label] = {
            "frequencies_hz": frequencies,
            "gain": gains,
            "phase_rad": phases,
        }

    figure, (gain_axis, phase_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8, 7),
    )
    for label, values in results.items():
        frequency = np.asarray(values["frequencies_hz"])
        gain = np.asarray(values["gain"])
        phase = np.asarray(values["phase_rad"])
        gain_axis.semilogx(
            frequency,
            20.0 * np.log10(gain),
            "o-",
            label=label,
        )
        phase_axis.semilogx(
            frequency,
            np.degrees(phase),
            "o-",
            label=label,
        )
    gain_axis.set_ylabel("gain / dB")
    phase_axis.set_ylabel("phase lag / deg")
    phase_axis.set_xlabel("frequency / Hz")
    gain_axis.grid(True, which="both", alpha=0.3)
    phase_axis.grid(True, which="both", alpha=0.3)
    gain_axis.legend()
    phase_axis.legend()
    figure.suptitle(f"J{joint} frequency response")
    figure.tight_layout()
    figure.savefig(args.output_dir / f"bode_J{joint}.png", dpi=160)
    plt.close(figure)

    report = args.output_dir / f"sweep_metrics_J{joint}.json"
    report.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
