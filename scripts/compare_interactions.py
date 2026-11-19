"""Compare controlled evaluations of a single checkpoint; no simulator imports."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def values(rows, key):
    return np.array([float(row[key]) for row in rows])


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
        path = Path(directory)
        summary = json.loads((path / "summary.json").read_text())
        body = summary["interaction"]["body"]
        rows = [row for row in read_csv(path / "bodies.csv") if row["body"] == body]
        steps = read_csv(path / "steps.csv")
        if not rows:
            parser.error(f"No samples for {body} in {path}")
        label = args.labels[i] if args.labels else path.name
        runs.append((label, summary, rows, steps))
    protocols = [summary.get("protocol") for _, summary, _, _ in runs]
    if any(protocol != protocols[0] for protocol in protocols):
        parser.error("Runs use different datasets, motion IDs, starting frames, time steps, checkpoints or seeds")
    bodies = [summary["interaction"]["body"] for _, summary, _, _ in runs]
    if len(set(bodies)) != 1:
        parser.error("Runs must use the same interaction body")
    timing = [(s["interaction"]["start"], s["interaction"]["end"], s["interaction"]["ramp"]) for _, s, _, _ in runs]
    if any(item != timing[0] for item in timing):
        parser.error("Runs must use the same force start/end/ramp times")
    baselines = [rows for _, summary, rows, _ in runs if summary["interaction"]["mode"] == "none"]
    baseline = baselines[0] if baselines else None
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    axes = axes.ravel()
    comparison = []
    for label, summary, rows, steps in runs:
        time = values(rows, "time_s")
        params = summary["interaction"]
        is_load = params["mode"] == "load"
        force_key = "load_mag_N" if is_load else "spring_mag_N"
        curve, = axes[0].plot(time, values(rows, force_key), label=label)
        color = curve.get_color()
        if not is_load and params["mode"] != "none":
            axes[0].plot(time, values(rows, "safe_limit_N"), "--", color=color, alpha=.5)
        yielding = values(rows, "yielding_m")
        if baseline is not None:
            # Recompute the no-load baseline displacement along THIS run's direction.
            direction = np.array([0., 0., -1.]) if is_load else np.array(params["direction"], dtype=float)
            direction /= np.linalg.norm(direction)
            baseline_displacement = np.stack([values(baseline, f"actual_{axis}_m")-values(baseline, f"reference_{axis}_m") for axis in "xyz"], axis=1) @ direction
            baseline_times = values(baseline, "time_s")
            yielding -= np.interp(time, baseline_times, baseline_displacement, left=np.nan, right=np.nan)
        axes[1].plot(time, 100*yielding, label=label, color=color)
        axes[2].plot(time, 100*values(rows, "tracking_error_m"), label=label, color=color)
        axes[3].plot(time, 100*values(rows, "root_error_m"), label=label, color=color)
        axes[4].plot(values(steps, "time_s"), values(steps, "arm_peak_torque_Nm"), label=label, color=color)
        axes[5].plot(time, values(rows, "torso_tilt_deg"), label=label, color=color)
        for phase, metrics in summary["phases"].items():
            comparison.append(dict(run=label, phase=phase, terminated=summary["terminated"],
                                   duration_s=summary["recorded_duration_s"], **metrics))
        print(f"{label}: {summary['recorded_duration_s']:.2f}s, terminated={summary['terminated']}; phases={list(summary['phases'])}")
    titles = ["Applied spring / payload force (N); dashed: safety input",
              "Displacement along force (cm)" + (" minus unloaded baseline" if baseline else " relative to original motion"),
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
