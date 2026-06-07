import json
import os
import pandas as pd

generations = [
    "archive/gen3/gen3_all_data.json",
    "archive/gen4/gen4_all_data.json",
    "archive/gen18/gen18_all_data.json",
    "archive/gen_robust_iter3/gen_robust_iter3_all_data.json",
    "archive/gen_robust_iter7/gen_robust_iter7_all_data.json",
    "archive/gen20/gen20_all_data.json",
]


def get_param(params, key, default="N/A"):
    return params.get(key, default)


results = []

for gen_file in generations:
    if not os.path.exists(gen_file):
        print(f"Skipping {gen_file} (not found)")
        continue

    try:
        with open(gen_file, "r") as f:
            data = json.load(f)

        runs = data.get("runs", [])

        for run in runs:
            # Filter only finished runs or interesting ones if needed
            # For now, let's grab all finished runs to be safe
            if run.get("status") != "FINISHED":
                continue

            params = run.get("all_parameters", {})
            metrics = run.get("performance", {}).get("best_metrics", {})
            analysis = run.get("analysis", {})

            # Extract key info
            run_name = run.get("run_name", "Unknown")

            # LR
            max_lr = get_param(
                params,
                "scheduler_opts/args/max_lr",
                get_param(params, "optim_opts/args/lr", "N/A"),
            )

            # Augs
            h_flip = get_param(params, "loader_opts/augmentations/h_flip_prob", "N/A")
            v_flip = get_param(params, "loader_opts/augmentations/v_flip_prob", "N/A")
            rot_prob = get_param(params, "loader_opts/augmentations/rotate_prob", "N/A")
            rot_limit = get_param(
                params, "loader_opts/augmentations/rotate_limit", "N/A"
            )

            # Loss
            cw = get_param(params, "loss_opts/class_weights", "N/A")

            # Metrics
            dci_iou = metrics.get("best_test_Debris_iou", 0.0)
            ci_iou = metrics.get("best_test_CleanIce_iou", 0.0)

            # Determine generation label
            gen_label = os.path.basename(os.path.dirname(gen_file))

            results.append(
                {
                    "Generation": gen_label,
                    "Run Name": run_name,
                    "Max LR": max_lr,
                    "H Flip": h_flip,
                    "Rot Limit": rot_limit,
                    "Class Weights": cw,
                    "DCI IoU": round(dci_iou, 4),
                    "Overfitting": analysis.get("overfitting_indicator", False),
                }
            )

    except Exception as e:
        print(f"Error reading {gen_file}: {e}")

# Create DataFrame and sort by DCI IoU desc within generation
df = pd.DataFrame(results)
if df.empty:
    print("No finished runs found in configured generation archives.")
    raise SystemExit(0)

# Sort primarily by Generation order, then by DCI IoU
gen_order = {
    "gen3": 1,
    "gen4": 2,
    "gen18": 3,
    "gen_robust_iter3": 4,
    "gen_robust_iter7": 5,
    "gen20": 6,
}
df["GenOrder"] = df["Generation"].map(gen_order)
df = df.sort_values(by=["GenOrder", "DCI IoU"], ascending=[True, False])

# Select top run for each generation + specific interesting ones
print(
    df[
        [
            "Generation",
            "Run Name",
            "Max LR",
            "H Flip",
            "Rot Limit",
            "Class Weights",
            "DCI IoU",
            "Overfitting",
        ]
    ].to_string(index=False)
)
