"""Fit JGarm hardware-profile parameters from offline samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jgarm_sim.identification import (  # noqa: E402
    identify_parameters,
    load_csv,
    write_profile_yaml,
    write_report,
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "identified_hardware_profile.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV with time_s, q1..q6, dq1..dq6, and tau1..tau6.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Generated ROS parameter YAML.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON diagnostics; defaults to the output stem with .json.",
    )
    parser.add_argument(
        "--profile-id",
        default="identified",
        help="Value written to hardware_profile.id.",
    )
    parser.add_argument(
        "--j5-config-weight",
        type=float,
        default=0.35,
        help="J5 q2 coupling weight used before fitting.",
    )
    parser.add_argument(
        "--non-conservative-j5",
        action="store_true",
        help=(
            "Fit the original reATRM J5 coupling instead of the "
            "projected form."
        ),
    )
    parser.add_argument(
        "--use-actual-velocity",
        action="store_true",
        help="Fit friction from measured velocity instead of target velocity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.j5_config_weight) or args.j5_config_weight <= 0.0:
        raise SystemExit("--j5-config-weight must be positive")
    samples = load_csv(args.csv)
    result = identify_parameters(
        samples,
        use_target_velocity=not args.use_actual_velocity,
        j5_config_weight=args.j5_config_weight,
        j5_conservative_coupling=not args.non_conservative_j5,
    )
    report_path = (
        args.report
        if args.report is not None
        else args.output.with_suffix(".json")
    )
    write_profile_yaml(
        args.output,
        result,
        profile_id=args.profile_id,
    )
    write_report(report_path, result)
    print(f"samples: {samples.time_s.size}")
    print(
        "gravity p: "
        + np.array2string(result.gravity.p_nm, precision=4)
    )
    print(
        "gravity q: "
        + np.array2string(result.gravity.q_nm, precision=4)
    )
    print(
        "gravity bias: "
        + np.array2string(result.gravity.bias_nm, precision=4)
    )
    print(
        "friction torque: "
        + np.array2string(result.friction.torque_nm, precision=4)
    )
    print(
        "friction scale: "
        + np.array2string(
            result.friction.velocity_scale_rad_s,
            precision=4,
        )
    )
    print(
        "breakaway torque: "
        + np.array2string(result.breakaway.torque_nm, precision=4)
    )
    print(
        "breakaway identified: "
        + str(result.breakaway.identified.tolist())
    )
    print(f"profile: {args.output}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
