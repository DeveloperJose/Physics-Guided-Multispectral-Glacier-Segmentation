#!/bin/bash

# Gen Robust Iter8 - Restoring Baseline & Optimizing Velocity Loss
# Servers: Frodo (4x 2080Ti), Bilbo (2x 3090), Desktop (1x 3060Ti)

# Define experiments
declare -A experiments

# Frodo Experiments (4 GPUs)
experiments["frodo:0"]="configs/frodo/debris_ice/gen_robust_iter8_dci_control_repro.yaml"
experiments["frodo:1"]="configs/frodo/debris_ice/gen_robust_iter8_dci_linear_vloss_full.yaml"
experiments["frodo:2"]="configs/frodo/debris_ice/gen_robust_iter8_dci_refined_selection.yaml"
experiments["frodo:3"]="configs/frodo/debris_ice/gen_robust_iter8_dci_linear_vloss_strong.yaml"

# Bilbo Experiments (2 GPUs)
experiments["bilbo:0"]="configs/bilbo/debris_ice/gen_robust_iter8_dci_vloss_thresh_1.yaml"
experiments["bilbo:1"]="configs/bilbo/debris_ice/gen_robust_iter8_dci_vloss_thresh_15.yaml"

# Desktop Experiments (1 GPU)
experiments["desktop:0"]="configs/desktop/debris_ice/gen_robust_iter8_dci_linear_landsat_only.yaml"

# Iterate and run based on current hostname
current_server=$(hostname)

echo "Starting sequential training on $current_server..."

for key in "${!experiments[@]}"; do
    server=${key%%:*}
    gpu=${key##*:}
    config=${experiments[$key]}

    if [ "$server" == "$current_server" ]; then
        echo "----------------------------------------------------------------"
        echo "Running experiment on $server GPU $gpu"
        echo "Config: $config"
        echo "----------------------------------------------------------------"
        
        uv run python scripts/train.py --config "$config" --server "$server" --gpu "$gpu"
        
        if [ $? -eq 0 ]; then
            echo "Successfully finished: $config"
        else
            echo "FAILED: $config"
        fi
        
        # Small cooldown between runs
        sleep 10
    fi
done

echo "All scheduled experiments for $current_server completed."
