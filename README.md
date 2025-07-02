# Neural Network Coding Layer (NNCL)

This repository contains the official PyTorch implementation for the paper: **"Neural Network Coding Layer (NNCL): Enhancing Deep Learning Robustness via Feature Restoration"**.
NNCL is a modular layer inspired by linear coding theory. It introduces structured redundancy into feature vectors, enabling the algebraic restoration of features lost to perturbations like dropout or noise. This enhances model robustness while retaining the regularization benefits of feature erasure.

## Features

- **NNCL Module**: A PyTorch `nn.Module` that can be easily integrated into various deep learning architectures.
- **Robustness Evaluation**: Scripts to reproduce experiments on CIFAR-10/100 with ResNet, EfficientNet, and Vision Transformer backbones.
- **Extensible**: Designed to be flexible for integration into new models and tasks.


## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/nncl-paper.git](https://github.com/your-username/nncl-paper.git)
    cd nncl-paper
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. You can generate one from your environment using `pip freeze > requirements.txt`)*

---

## Reproducing the Results

### Reproducing All Paper Results

All experiments are defined and launched from `launcher.py`. This script calls `nncl_experiment.py` with the appropriate arguments for each experimental setting described in the paper.

To run all experiments and reproduce the paper's main results, simply execute the launcher script:
```bash
python launcher.py

This will run all variants (Baseline, NNCL-Fixed, NNCL-Learnable) for all backbone-dataset combinations, as well as the ablation studies for learning rate and redundancy levels.

 - Logs: A detailed log file for each experiment will be saved in the results/ directory (e.g., results/c10_r50_nncl_learnA.log).
 - Results: A summary CSV for each run will be saved in results/ (e.g., results/c10_r50_nncl_learnA.csv).
 - Learning Curves: Per-epoch training curves are saved in results/curves/.

Running a Single Experiment
You can also run a single experiment directly using nncl_experiment.py. For example, to train a ResNet-50 model on CIFAR-10 with a learnable NNCL:

python nncl_experiment.py \
    --dataset cifar10 \
    --backbones resnet50 \
    --epochs 120 \
    --batch 128 \
    --warmup_epochs 5 \
    --lr 1e-4 \
    --dropout_rates 0.2 \
    --noise_stds 0.05 \
    --erasure_rates 0.0 0.2 0.4 0.6 \
    --redundancy 0.5 \
    --learnable_A \
    --aux_rec_loss \
    --lambda_rec 1.0 \
    --csv_out results/my_single_run.csv


Acknowledgements
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS-2024-00459703).
