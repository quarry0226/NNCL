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
import time

PYTHON = "python -u"
SCRIPT = "nncl_experiment.py" # Ensure this matches your experiment script name
RESULTS_DIR = "results"

# Common arguments for all experiments
COMMON_ARGS = [
    "--epochs", "120",
    "--num_workers", "8",
    "--warmup_epochs", "5"
]

# Flags for NNCL variants
NNCL_FLAGS_LEARNABLE = ["--learnable_A", "--aux_rec_loss", "--lambda_rec", "1.0", "--redundancy", "0.5", "--use_masked_var"]
NNCL_FLAGS_FIXED = ["--aux_rec_loss", "--lambda_rec", "1.0", "--redundancy", "0.5", "--use_masked_var"]

# --- Experiment Definitions ---
EXPERIMENTS = []
dp_shared = "0.2"
noise_shared = "0.05"
erasures_shared = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]

# Sanity check
EXPERIMENTS.append(("sanity_check", ["--dataset", "cifar10", "--backbones", "resnet18", "--epochs", "5", "--batch", "128", "--noise_stds", noise_shared]))

def add(name, args):
    EXPERIMENTS.append((name, args + COMMON_ARGS))

def add_nncl_variants(tag, dataset, backbone):
    add(f"{tag}_nncl_learnA", ["--dataset", dataset, "--backbones", backbone, "--dropout_rates", dp_shared, "--noise_stds", noise_shared, "--erasure_rates", *erasures_shared, "--batch", "128", *NNCL_FLAGS_LEARNABLE])
    add(f"{tag}_nncl_fixedA", ["--dataset", dataset, "--backbones", backbone, "--dropout_rates", dp_shared, "--noise_stds", noise_shared, "--erasure_rates", *erasures_shared, "--batch", "128", *NNCL_FLAGS_FIXED])

# CIFAR-10 / ResNet-50
add_nncl_variants("c10_r50", "cifar10", "resnet50")
add("c10_r50_base", ["--dataset", "cifar10", "--backbones", "resnet50", "--no_nncl", "--dropout_rates", dp_shared, "--noise_stds", noise_shared, "--erasure_rates", *erasures_shared, "--batch", "128"])

# CIFAR-100 / EfficientNet-B0
add_nncl_variants("c100_eff", "cifar100", "efficientnet_b0")
add("c100_eff_base", ["--dataset", "cifar100", "--backbones", "efficientnet_b0", "--no_nncl", "--dropout_rates", dp_shared, "--noise_stds", noise_shared, "--erasure_rates", "0", "0.2", "0.4", "0.5", "0.6", "--batch", "128"])

# # CIFAR-10 / ViT
# add_nncl_variants("c10_vit", "cifar10", "vit_base_patch16_224")
# add("c10_vit_base", ["--dataset", "cifar10", "--backbones", "vit_base_patch16_224", "--no_nncl", "--dropout_rates", dp_shared, "--noise_stds", noise_shared, "--erasure_rates", *erasures_shared, "--batch", "128"])

# Learning Rate Tuning
learning_rates = ["1e-4", "3e-4", "5e-5"]
for lr in learning_rates:
    for backbone in ["resnet50", "efficientnet_b0"]:
        dataset = "cifar100" if backbone == "efficientnet_b0" else "cifar10"
        add(f"lr_tune_{backbone}_lr{lr.replace('-', '')}_learnA", ["--dataset", dataset, "--backbones", backbone, "--dropout_rates", dp_shared, "--noise_stds", noise_shared, "--erasure_rates", "0", "0.2", "0.4", "--batch", "128", "--lr", lr, *NNCL_FLAGS_LEARNABLE])
        add(f"lr_tune_{backbone}_lr{lr.replace('-', '')}_fixedA", ["--dataset", dataset, "--backbones", backbone, "--dropout_rates", dp_shared, "--noise_stds", noise_shared, "--erasure_rates", "0", "0.2", "0.4", "--batch", "128", "--lr", lr, *NNCL_FLAGS_FIXED])

# Redundancy Tuning
redundancy_values = ["0.05", "0.1", "0.25", "0.5", "0.75"]
for redundancy in redundancy_values:
    add(f"redundancy_tune_r50_k{redundancy.replace('.', '')}", ["--dataset", "cifar10", "--backbones", "resnet50", "--dropout_rates", dp_shared, "--noise_stds", noise_shared, "--erasure_rates", "0", "0.2", "0.4", "--batch", "128", "--redundancy", redundancy, *NNCL_FLAGS_LEARNABLE])

# --- Execution Loop ---
os.makedirs(RESULTS_DIR, exist_ok=True)

for idx, (tag, args_list) in enumerate(EXPERIMENTS, 1):
    csv_path = os.path.join(RESULTS_DIR, f"{tag}.csv")
    log_path = os.path.join(RESULTS_DIR, f"{tag}.log")
    cmd = [*PYTHON.split(), SCRIPT, *args_list, "--csv_out", csv_path]

    print(f"\n[Run {idx}/{len(EXPERIMENTS)}] Starting: {tag}")
    print("Command:", " ".join(cmd))

    start_time = time.monotonic()

    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )

            # Stream stdout to both console and log file
            for line in proc.stdout:
                print(line, end="")
                logf.write(line)

            proc.wait()
    finally:
        end_time = time.monotonic()
        duration = end_time - start_time
        
        # Format duration into h, m, s
        minutes, seconds = divmod(duration, 60)
        hours, minutes = divmod(minutes, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        
        # Check process exit code and print final status
        if proc.returncode == 0:
            status_msg = f"[{tag}] Finished (Success) | Elapsed Time: {time_str}"
            print(status_msg)
        else:
            status_msg = f"[{tag}] Finished (Error code: {proc.returncode}) | Elapsed Time: {time_str}"
            print(status_msg)
            print(f"See log for details -> {log_path}")

        # Append final status to the log file
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"\n--- EXECUTION FINISHED ---\n")
            logf.write(f"Status: {'Success' if proc.returncode == 0 else f'Failed with code {proc.returncode}'}\n")
            logf.write(f"Total Execution Time: {time_str}\n")
