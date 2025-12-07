import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys


def get_latest_run_dir(base_dir="output"):
    runs = glob.glob(os.path.join(base_dir, "*"))
    runs.sort(key=os.path.getmtime, reverse=True)
    if not runs:
        return None
    return runs[0]


def inspect_sigmas(run_dir):
    print(f"Inspecting run: {run_dir}")
    # Find event file
    # Try multiple locations
    potential_paths = [
        os.path.join(run_dir, "version_0", "events.out.tfevents.*"),
        os.path.join(run_dir, "logs", "version_0", "events.out.tfevents.*"),
        os.path.join(run_dir, "events.out.tfevents.*"),
    ]

    event_files = []
    for p in potential_paths:
        found = glob.glob(p)
        if found:
            event_files.extend(found)

    if not event_files:
        print("No event files found.")
        return

    event_file = event_files[0]
    print(f"Reading event file: {event_file}")

    ea = EventAccumulator(event_file)
    ea.Reload()

    tags = ea.Tags()["scalars"]

    sigma_tags = [t for t in tags if "sigma" in t]
    print(f"Found sigma tags: {sigma_tags}")

    for tag in sigma_tags:
        events = ea.Scalars(tag)
        if events:
            # Print first, middle, and last value
            values = [e.value for e in events]
            steps = [e.step for e in events]
            print(f"\n{tag}:")
            print(f"  Start (step {steps[0]}): {values[0]:.4f}")
            if len(values) > 1:
                print(f"  End   (step {steps[-1]}): {values[-1]:.4f}")
                print(f"  Change: {values[-1] - values[0]:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = get_latest_run_dir()

    if run_dir:
        inspect_sigmas(run_dir)
    else:
        print("No run directory found.")
