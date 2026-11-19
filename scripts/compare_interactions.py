"""Compare controlled evaluations of a single checkpoint; no simulator imports."""

import argparse
import csv
import difflib
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open() as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def values(rows, key):
    return np.array([float(row[key]) for row in rows])


def describe_missing_run(path):
    required = ("summary.json", "bodies.csv", "steps.csv")
    missing = [name for name in required if not (path / name).is_file()]
    lines = [f"Run directory is missing required files: {path}",
             f"  Missing: {', '.join(missing)}"]
    parent = path.parent
    if parent.is_dir():
        candidates = sorted(child for child in parent.iterdir()
                            if child.is_dir() and (child / "summary.json").is_file())
        close = difflib.get_close_matches(path.name, [child.name for child in candidates], n=5, cutoff=0.35)
        if close:
            lines.append("  Existing runs in the same parent:")
            lines.extend(f"    {parent / name}" for name in close)
        elif candidates:
            lines.append("  Run directories with summary.json in the same parent:")
            lines.extend(f"    {child}" for child in candidates[:10])
    lines.append("Pass each experiment output directory (the one containing summary.json), not the comparison output directory.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Experiment output directories.")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", default="scripts/outputs_eval/comparison")
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.runs):
        parser.error("--labels must have one entry per run")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for i, directory in enumerate(args.runs):
        path = Path(directory).expanduser().resolve()
        required = ("summary.json", "bodies.csv", "steps.csv")
        if not path.is_dir() or any(not (path / name).is_file() for name in required):
            parser.error(describe_missing_run(path))
        try:
            summary = json.loads((path / "summary.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"Could not read {path / 'summary.json'}: {exc}")
        if "interaction" not in summary or "body" not in summary["interaction"]:
            parser.error(f"Invalid summary schema in {path / 'summary.json'}: expected interaction.body")
        body = summary["interaction"]["body"]
        body_columns, body_data = read_csv(path / "bodies.csv")
        step_columns, steps = read_csv(path / "steps.csv")
        if "body" not in body_columns:
            parser.error(f"{path / 'bodies.csv'} has no 'body' column; found: {body_columns}")
        rows = [row for row in body_data if row["body"] == body]
        if not rows:
            parser.error(f"No samples for {body} in {path}")
        missing_body_columns = sorted({"time_s", "yielding_m", "tracking_error_m", "root_error_m", "torso_tilt_deg"} - set(body_columns))
        missing_step_columns = sorted({"time_s", "arm_peak_torque_Nm"} - set(step_columns))
        if missing_body_columns or missing_step_columns:
            details = []
            if missing_body_columns:
                details.append(f"bodies.csv missing {missing_body_columns}")
            if missing_step_columns:
                details.append(f"steps.csv missing {missing_step_columns}")
            parser.error(f"Incompatible CSV schema in {path}: {'; '.join(details)}")
        label = args.labels[i] if args.labels else path.name
        runs.append((label, summary, rows, steps, set(body_columns)))
    protocols = [summary.get("protocol") for _, summary, _, _, _ in runs]
    if any(protocol != protocols[0] for protocol in protocols):
        parser.error("Runs use different datasets, motion IDs, starting frames, time steps, checkpoints or seeds")
    bodies = [summary["interaction"]["body"] for _, summary, _, _, _ in runs]
    if len(set(bodies)) != 1:
        parser.error("Runs must use the same interaction body")
    timing = [(s["interaction"]["start"], s["interaction"]["end"], s["interaction"]["ramp"]) for _, s, _, _, _ in runs]
    if any(item != timing[0] for item in timing):
        parser.error("Runs must use the same force start/end/ramp times")
    baselines = [(summary, rows, columns) for _, summary, rows, _, columns in runs if summary["interaction"]["mode"] == "none"]
    if len(baselines) > 1:
        parser.error("Pass at most one --experiment none baseline")
    baseline = baselines[0][1] if baselines else None
    baseline_columns = baselines[0][2] if baselines else set()
    has_reference_yielding = all("reference_yielding_m" in columns for _, _, _, _, columns in runs)
    if not has_reference_yielding:
        print("[Warn] At least one bodies.csv lacks reference_yielding_m; plotting actual yield only (solid lines).")
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    axes = axes.ravel()
    comparison = []
    for label, summary, rows, steps, columns in runs:
        time = values(rows, "time_s")
        params = summary["interaction"]
        is_external = params["mode"] == "load"
        force_key = ("load_mag_N" if "load_mag_N" in columns else "external_force_mag_N") if is_external else "spring_mag_N"
        if force_key not in columns and params["mode"] == "force":
            force_key = "total_mag_N" if "total_mag_N" in columns else "spring_mag_N"
        if force_key not in columns:
            parser.error(f"{label}: bodies.csv has no force column for mode {params['mode']}; found: {sorted(columns)}")
        curve, = axes[0].plot(time, values(rows, force_key), label=label)
        color = curve.get_color()
        if params["mode"] == "spring":
            axes[0].plot(time, values(rows, "safe_limit_N"), "--", color=color, alpha=.5)
        yielding = values(rows, "yielding_m")
        if baseline is not None:
            # Recompute the no-load baseline displacement along THIS run's direction.
            direction = (np.array(params["force_vector"], dtype=float) if params["mode"] == "force"
                         else np.array([0., 0., -1.]) if params["mode"] == "load"
                         else np.array(params["direction"], dtype=float))
            direction /= np.linalg.norm(direction)
            baseline_displacement = np.stack([values(baseline, f"actual_{axis}_m")-values(baseline, f"reference_{axis}_m") for axis in "xyz"], axis=1) @ direction
            baseline_times = values(baseline, "time_s")
            yielding -= np.interp(time, baseline_times, baseline_displacement, left=np.nan, right=np.nan)
        axes[1].plot(time, 100*yielding, label=f"{label} actual", color=color)
        if has_reference_yielding:
            ref_yielding = values(rows, "reference_yielding_m")
            if baseline is not None and "reference_yielding_m" in baseline_columns:
                ref_yielding -= np.interp(time, values(baseline, "time_s"),
                                          values(baseline, "reference_yielding_m"),
                                          left=np.nan, right=np.nan)
            axes[1].plot(time, 100*ref_yielding, label=f"{label} compliant reference",
                         color=color, linestyle="--", alpha=.8)
        axes[2].plot(time, 100*values(rows, "tracking_error_m"), label=label, color=color)
        axes[3].plot(time, 100*values(rows, "root_error_m"), label=label, color=color)
        axes[4].plot(values(steps, "time_s"), values(steps, "arm_peak_torque_Nm"), label=label, color=color)
        axes[5].plot(time, values(rows, "torso_tilt_deg"), label=label, color=color)
        for phase, metrics in summary["phases"].items():
            comparison.append(dict(run=label, phase=phase, terminated=summary["terminated"],
                                   duration_s=summary["recorded_duration_s"], **metrics))
        print(f"{label}: {summary['recorded_duration_s']:.2f}s, terminated={summary['terminated']}; phases={list(summary['phases'])}")
    titles = ["Applied spring / fixed external force (N); dashed: safety input",
              "Force-direction yield (cm); solid: robot, dashed: compliant reference" + (" minus unloaded baseline" if baseline else " relative to original motion"),
              "Selected link tracking error vs original motion (cm)",
              "Root position error vs original motion (cm)",
              "Peak absolute arm joint actuator torque (Nm)", "Torso tilt from vertical (deg)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        ax.set_xlabel("Simulation time (s)")
        p = runs[0][1]["interaction"]
        ax.axvspan(p["start"], p["end"], color="grey", alpha=.08)
        ax.axvline(p["start"], color="grey", linestyle=":", linewidth=.8)
        ax.axvline(p["end"], color="grey", linestyle=":", linewidth=.8)
    fig.suptitle(f"Single-policy controlled interaction: {bodies[0]}")
    fig.tight_layout()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print(f"Saved {output.resolve()}/comparison.png, comparison.pdf, comparison.csv")


if __name__ == "__main__":
    main()
