#!/usr/bin/env python3
"""
NNCL Experiment Launcher
========================
This script orchestrates the execution of all experiments required to reproduce
the results in the paper "Neural Network Coding Layer (NNCL): Enhancing Deep
Learning Robustness via Feature Restoration".

It defines a series of experimental configurations and runs them sequentially
using `nncl_experiment.py`.

author = Lee Chae-Seok
email = "quarry@kaist.ac.kr"
"""

import os
import subprocess

# --- Configuration ---
PYTHON_EXECUTABLE = "python"  # Or "python3" depending on your system
MAIN_SCRIPT = "nncl_experiment.py"
RESULTS_DIR = "results"

# Common arguments applied to all experiments
COMMON_ARGS = [
    "--epochs", "120",
    "--batch", "128",
    "--num_workers", "8",
    "--warmup_epochs", "5",
    "--lr", "1e-4"
]

# Base flags for NNCL variants. Tunable parameters like redundancy are excluded here.
NNCL_FLAGS_LEARNABLE_BASE = ["--learnable_A", "--aux_rec_loss", "--lambda_rec", "1.0", "--use_masked_var"]
NNCL_FLAGS_FIXED_BASE = ["--use_masked_var"] # Fixed matrix does not need aux loss by default

# Shared parameters for main experiments
DP_RATE_SHARED = "0.2"
NOISE_STD_SHARED = "0.05"
ERASURE_RATES_SHARED = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]

# --- Experiment Definitions ---
EXPERIMENTS = []

def add_experiment(name, args):
    """Helper function to add an experiment to the list."""
    # Combine specific experiment args with common args
    full_args = args + COMMON_ARGS
    EXPERIMENTS.append((name, full_args))

# --- 1. Main Experiments (NNCL vs. Baseline) ---

# a) CIFAR-10 / ResNet-50
add_experiment("c10_r50_nncl_learnA", ["--dataset", "cifar10", "--backbones", "resnet50", "--dropout_rates", DP_RATE_SHARED, "--noise_stds", NOISE_STD_SHARED, "--erasure_rates", *ERASURE_RATES_SHARED, "--redundancy", "0.5", *NNCL_FLAGS_LEARNABLE_BASE])
add_experiment("c10_r50_nncl_fixedA", ["--dataset", "cifar10", "--backbones", "resnet50", "--dropout_rates", DP_RATE_SHARED, "--noise_stds", NOISE_STD_SHARED, "--erasure_rates", *ERASURE_RATES_SHARED, "--redundancy", "0.5", *NNCL_FLAGS_FIXED_BASE])
add_experiment("c10_r50_baseline",    ["--dataset", "cifar10", "--backbones", "resnet50", "--no_nncl", "--dropout_rates", DP_RATE_SHARED, "--noise_stds", NOISE_STD_SHARED, "--erasure_rates", *ERASURE_RATES_SHARED])

# b) CIFAR-100 / EfficientNet-B0
add_experiment("c100_eff_nncl_learnA", ["--dataset", "cifar100", "--backbones", "efficientnet_b0", "--dropout_rates", DP_RATE_SHARED, "--noise_stds", NOISE_STD_SHARED, "--erasure_rates", *ERASURE_RATES_SHARED, "--redundancy", "0.5", *NNCL_FLAGS_LEARNABLE_BASE])
add_experiment("c100_eff_nncl_fixedA", ["--dataset", "cifar100", "--backbones", "efficientnet_b0", "--dropout_rates", DP_RATE_SHARED, "--noise_stds", NOISE_STD_SHARED, "--erasure_rates", *ERASURE_RATES_SHARED, "--redundancy", "0.5", *NNCL_FLAGS_FIXED_BASE])
add_experiment("c100_eff_baseline",    ["--dataset", "cifar100", "--backbones", "efficientnet_b0", "--no_nncl", "--dropout_rates", DP_RATE_SHARED, "--noise_stds", NOISE_STD_SHARED, "--erasure_rates", *ERASURE_RATES_SHARED])

# c) CIFAR-10 / Vision Transformer (Future Work)
# add_experiment("c10_vit_nncl_learnA", ["--dataset", "cifar10", "--backbones", "vit_base_patch16_224", "--dropout_rates", DP_RATE_SHARED, "--noise_stds", NOISE_STD_SHARED, "--erasure_rates", *ERASURE_RATES_SHARED, "--redundancy", "0.5", *NNCL_FLAGS_LEARNABLE_BASE])
# add_experiment("c10_vit_nncl_fixedA", ["--dataset", "cifar10", "--backbones", "vit_base_patch16_224", "--dropout_rates", DP_RATE_SHARED, "--noise_stds", NOISE_STD_SHARED, "--erasure_rates", *ERASURE_RATES_SHARED, "--redundancy", "0.5", *NNCL_FLAGS_FIXED_BASE])
# add_experiment("c10_vit_baseline",    ["--dataset", "cifar10", "--backbones", "vit_base_patch16_224", "--no_nncl", "--dropout_rates", DP_RATE_SHARED, "--noise_stds", NOISE_STD_SHARED, "--erasure_rates", *ERASURE_RATES_SHARED])


# --- 2. Ablation Studies ---

# a) Redundancy Level Tuning (on ResNet-50)
redundancy_values = ["0.05", "0.1", "0.25", "0.5", "0.75"]
for k in redundancy_values:
    # Use LEARNABLE_BASE flags and add the specific redundancy `k` for this run
    add_experiment(f"ablation_redundancy_k{k.replace('.', '')}", ["--dataset", "cifar10", "--backbones", "resnet50", "--dropout_rates", DP_RATE_SHARED, "--noise_stds", NOISE_STD_SHARED, "--erasure_rates", "0.0", "0.2", "0.4", "--redundancy", k, *NNCL_FLAGS_LEARNABLE_BASE])

# Note: Learning rate tuning experiments can be added here in a similar fashion if needed.
# For example:
# learning_rates = ["3e-4", "5e-5"]
# for lr in learning_rates:
#     add_experiment(f"ablation_lr_{lr}", ["--dataset", "cifar10", "--backbones", "resnet50", "--lr", lr, ...])


# --- Execution ---

def main():
    """Iterates through and runs all defined experiments."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    total_experiments = len(EXPERIMENTS)

    for idx, (tag, args_list) in enumerate(EXPERIMENTS, 1):
        # Define paths for log and result files for this specific run
        csv_path = os.path.join(RESULTS_DIR, f"{tag}_summary.csv")
        log_path = os.path.join(RESULTS_DIR, f"{tag}.log")
        
        # Construct the full command
        cmd = [PYTHON_EXECUTABLE, "-u", MAIN_SCRIPT, *args_list, "--csv_out", csv_path]

        print(f"\n{'='*25}")
        print(f"▶️ Running Experiment {idx}/{total_experiments}: {tag}")
        print(f"{'='*25}")
        print(f"📄 Command: {' '.join(cmd)}")

        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                # Use Popen to capture output in real-time
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                )

                # Stream output to both console and log file
                for line in process.stdout:
                    print(line, end="")
                    log_file.write(line)
                
                process.wait()

            if process.returncode == 0:
                print(f"\n✅ [{tag}] Succeeded.")
            else:
                print(f"\n❌ [{tag}] Failed with error code {process.returncode}.")
                print(f"   Check the log file for details: {log_path}")

        except Exception as e:
            print(f"\n❌ An exception occurred while running '{tag}': {e}")
            print(f"   Check the log file for details: {log_path}")

if __name__ == "__main__":
    main()
