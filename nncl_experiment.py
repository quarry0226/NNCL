#!/usr/bin/env python
"""
NNCL Experiment Runner
======================
This script runs experiments for the Neural Network Coding Layer (NNCL) to evaluate
its robustness against feature erasure on image classification tasks.

Key Features:
- Implements the NNCL module with a sample-wise restoration loop.
- Supports multiple backbone architectures (ResNet, EfficientNet, ViT).
- Handles training, evaluation, and logging of results.
- Configurable via command-line arguments for systematic experiments.

This version is intended for public release to reproduce the results from the paper.
The core logic matches the implementation used for the original experiments.

author = Lee Chae-Seok
email = "quarry@kaist.ac.kr"
"""

import argparse
import csv
import logging
import math
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet18, resnet50

try:
    import timm  # For Vision-Transformer backbone
except ImportError:
    timm = None

# --- Configuration ---
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

# --- Utilities ---

class GaussianNoise:
    """Adds i.i.d. Gaussian noise to a tensor in the [0,1] range."""
    def __init__(self, std: float):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std == 0:
            return x
        return torch.clamp(x + torch.randn_like(x) * self.std, 0, 1)

def set_seed(seed: int, deterministic: bool = False):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

def worker_init(worker_id: int, base_seed: int):
    """Initializes a data loader worker with a unique seed."""
    set_seed(base_seed + worker_id)

# --- NNCL Module ---

class NNCL(nn.Module):
    """
    Neural Network Coding Layer (NNCL).

    This implementation uses a sample-wise loop for feature restoration,
    matching the logic described in Algorithm 1 of the paper.
    """
    def __init__(self, d_in: int, k: float = 0.25, learnA: bool = False, use_masked_var: bool = False):
        super().__init__()
        self.use_masked_var = use_masked_var
        d_out = math.ceil(d_in * (1 + k))
        self.A = nn.Parameter(torch.empty(d_out, d_in), requires_grad=learnA)
        nn.init.orthogonal_(self.A)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: The original feature tensor (B, D_in).
            mask: The binary dropout mask (B, D_in), where 1 indicates an erased feature.

        Returns:
            x_rec: The reconstructed feature tensor (B, D_in).
            mse: A tensor of per-sample Mean Squared Error for erased features (B,).
            fr_acc: A tensor of per-sample Feature Restoration Accuracy (B,).
            num_erased: A tensor of the number of erased elements per sample (B,).
        """
        # 1. Linear Encoding
        y = F.linear(x, self.A)
        B, D = x.shape
        x_rec = x.clone()
        mask_f = mask.float()

        # 2. Per-sample feature restoration (iterative loop)
        for b in range(B):
            E = torch.nonzero(mask[b], as_tuple=False).view(-1)
            if E.numel() == 0:
                continue
            K = torch.nonzero(~mask[b], as_tuple=False).view(-1)

            A_E = self.A[:, E]  # (d_out, |E|)
            A_K = self.A[:, K]  # (d_out, |K|)
            
            # If all features are known, no restoration needed for this sample
            if A_K.numel() == 0: 
                continue

            # Compute Z = y - A_K @ x_K
            Z = y[b] - (A_K @ x[b, K])

            # Restore X_E using the pseudo-inverse
            try:
                A_E_pinv = torch.linalg.pinv(A_E)
                x_E_rec = A_E_pinv @ Z
                x_rec[b, E] = x_E_rec
            except torch.linalg.LinAlgError:
                # If pinv fails, keep erased features as 0
                pass
        
        # 3. Compute FR metrics
        num_erased = mask_f.sum(dim=1)
        # Avoid division by zero for samples with no erasure
        num_erased_clamped = torch.clamp(num_erased, min=1.0)

        # Per-sample Mean Squared Error (on erased elements only)
        diff_sq = (x_rec - x).pow(2) * mask_f
        mse = diff_sq.sum(dim=1) / num_erased_clamped

        # Per-sample Variance (either on erased elements or all elements)
        if self.use_masked_var:
            mean_masked = (x * mask_f).sum(dim=1, keepdim=True) / num_erased_clamped.unsqueeze(1)
            var = ((x - mean_masked).pow(2) * mask_f).sum(dim=1) / num_erased_clamped
        else:
            var = x.var(dim=1, unbiased=False)

        # FR Accuracy (based on ratio of standard deviations)
        epsilon = 1e-9
        relative_error = torch.sqrt(mse + epsilon) / torch.sqrt(var + epsilon)
        fr_acc = (1.0 - relative_error).clamp(0.0, 1.0)

        return x_rec, mse, fr_acc, num_erased


class Model(nn.Module):
    """A wrapper model that integrates a backbone with the NNCL module."""
    def __init__(self, name: str, n_cls: int, dp: float, k: float, learnA: bool, no_nncl: bool, use_masked_var: bool):
        super().__init__()
        name = name.lower()

        # --- Backbone Initialization ---
        if name == "resnet18":
            m = resnet18(weights="IMAGENET1K_V1")
            feat_dim = m.fc.in_features
            self.feat = nn.Sequential(*list(m.children())[:-1])
        elif name == "resnet50":
            m = resnet50(weights="IMAGENET1K_V2")
            feat_dim = m.fc.in_features
            self.feat = nn.Sequential(*list(m.children())[:-1])
        elif name == "efficientnet_b0":
            m = efficientnet_b0(weights="IMAGENET1K_V1")
            feat_dim = m.classifier[1].in_features
            self.feat = nn.Sequential(m.features, nn.AdaptiveAvgPool2d((1, 1)))
        elif name in {"vit_b_16", "vit_base_patch16_224"}:
            if timm is None:
                raise ImportError("Vision Transformer requires `pip install timm`.")
            m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
            feat_dim = m.num_features
            self.feat = m
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        self.cls = nn.Linear(feat_dim, n_cls)
        self.dp_p = dp
        
        # --- NNCL or Baseline (Identity) ---
        if no_nncl:
            self.nncl_module = nn.Identity()
        else:
            self.nncl_module = NNCL(feat_dim, k, learnA, use_masked_var)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor, eval_er: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.feat(x)
        if f.ndim > 2:
            f = f.flatten(1)

        # Determine erasure probability (dropout for train, erasure for eval)
        p = self.dp_p if self.training else eval_er
        mask = torch.rand_like(f) < p

        if isinstance(self.nncl_module, nn.Identity):
            f_use = f.masked_fill(mask, 0.0)
            # For baseline, metrics are placeholders as no restoration occurs
            B = f.size(0)
            dummy_zeros = torch.zeros(B, device=f.device)
            mse, fr_acc = dummy_zeros, dummy_zeros
            num_erased = mask.float().sum(dim=1)
        else:
            f_use, mse, fr_acc, num_erased = self.nncl_module(f, mask)

        logits = self.cls(f_use)
        return logits, mse, fr_acc, num_erased


# --- Training and Evaluation Logic ---

@torch.inference_mode()
def _metrics(preds, targets) -> Tuple[float, float]:
    """Computes accuracy and macro-F1 score."""
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return acc, f1

def run_epoch(model: nn.Module, loader, opt, scheduler, device, cfg, is_train: bool, eval_er: float = 0.0) -> Dict:
    """Runs a single epoch of training or evaluation."""
    model.train(is_train)
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss, total_erased, fr_weighted_sum = 0.0, 0, 0.0
    all_preds, all_targets = [], []
    
    actual_model = model.module if isinstance(model, nn.DataParallel) else model

    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        B = imgs.size(0)

        if is_train:
            opt.zero_grad(set_to_none=True)

        logits, mse_vec, fr_vec, num_erased_vec = model(imgs, eval_er)
        task_loss = ce_loss_fn(logits, lbls)

        if is_train:
            # Auxiliary loss is the mean squared error on erased features
            aux_loss = mse_vec.mean() if cfg.aux_rec_loss and not cfg.no_nncl else 0.0
            loss = task_loss + cfg.lambda_rec * aux_loss
            loss.backward()
            opt.step()
        else:
            loss = task_loss

        total_loss += loss.item() * B
        
        # Accumulate stats for element-wise FR accuracy
        if not cfg.no_nncl:
            erased_in_batch = num_erased_vec.sum().item()
            if erased_in_batch > 0:
                fr_weighted_sum += (fr_vec * num_erased_vec).sum().item()
                total_erased += int(erased_in_batch)

        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_targets.extend(lbls.cpu().tolist())

    if is_train:
        scheduler.step()

    n_samples = len(loader.dataset)
    acc, f1 = _metrics(all_preds, all_targets)
    avg_loss = total_loss / n_samples
    
    # Final element-wise average FR accuracy for the epoch
    fr_acc_epoch = fr_weighted_sum / total_erased if total_erased > 0 else 1.0

    return {"loss": avg_loss, "acc": acc, "f1": f1, "fr_acc": fr_acc_epoch}


# --- Experiment Orchestration ---

def get_dataloaders(cfg: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """Configures and returns data loaders for CIFAR-10/100."""
    is_vit = cfg.backbone.lower().startswith("vit")
    img_size = 224 if is_vit else 32
    
    # Normalization stats for CIFAR
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2023, 0.1994, 0.2010]

    train_transforms = [
        transforms.Resize((img_size, img_size)) if is_vit else nn.Identity(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        GaussianNoise(cfg.noise),
        transforms.Normalize(cifar_mean, cifar_std)
    ]
    
    test_transforms = [
        transforms.Resize((img_size, img_size)) if is_vit else nn.Identity(),
        transforms.ToTensor(),
        GaussianNoise(cfg.noise), # Apply same noise for consistency
        transforms.Normalize(cifar_mean, cifar_std)
    ]

    tf_train = transforms.Compose(train_transforms)
    tf_test = transforms.Compose(test_transforms)
    
    if cfg.dataset == "cifar10":
        Dataset = torchvision.datasets.CIFAR10
        n_cls = 10
    elif cfg.dataset == "cifar100":
        Dataset = torchvision.datasets.CIFAR100
        n_cls = 100
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    ds_train = Dataset(cfg.data_root, train=True, transform=tf_train, download=True)
    ds_test = Dataset(cfg.data_root, train=False, transform=tf_test, download=True)
    
    _worker_init_fn = partial(worker_init, base_seed=cfg.seed) if cfg.num_workers > 0 else None
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=cfg.batch, shuffle=True,
                                           num_workers=cfg.num_workers, pin_memory=True,
                                           worker_init_fn=_worker_init_fn, drop_last=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=cfg.batch, shuffle=False,
                                          num_workers=cfg.num_workers, pin_memory=True,
                                          worker_init_fn=_worker_init_fn)
    return dl_train, dl_test, n_cls


def run_single_experiment(cfg: argparse.Namespace, device: torch.device) -> List[Dict]:
    """Runs one full experiment: train, then evaluate at multiple erasure rates."""
    set_seed(cfg.seed, cfg.deterministic)
    
    # 1. Setup Dataloaders and Model
    dl_train, dl_test, n_cls = get_dataloaders(cfg)
    model = Model(
        name=cfg.backbone, n_cls=n_cls, dp=cfg.dp, k=cfg.redundancy, 
        learnA=cfg.learnable_A, no_nncl=cfg.no_nncl, use_masked_var=cfg.use_masked_var
    ).to(device)

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    # 2. Setup Optimizer and Scheduler
    lr = cfg.lr * (cfg.batch / 128) # Linear learning rate scaling
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return float(epoch + 1) / max(1, cfg.warmup_epochs)
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    logging.info(f"Optimizer: AdamW, LR: {lr:.2e}, Scheduler: {cfg.warmup_epochs}-epoch warmup with cosine decay.")

    # 3. Training Loop
    train_curve = []
    for ep in range(cfg.epochs):
        train_metrics = run_epoch(model, dl_train, opt, scheduler, device, cfg, is_train=True)
        val_metrics = run_epoch(model, dl_test, None, None, device, cfg, is_train=False, eval_er=0.0)
        
        train_curve.append({
            "epoch": ep + 1,
            "train_loss": train_metrics["loss"], "train_acc": train_metrics["acc"], "train_fr": train_metrics["fr_acc"],
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"], "val_fr": val_metrics["fr_acc"],
        })
        logging.info(f"Ep{ep+1:03d}/{cfg.epochs} | LR {opt.param_groups[0]['lr']:.2e} | "
                     f"TrL {train_metrics['loss']:.3f} TrAcc {train_metrics['acc']:.3f} TrFR {train_metrics['fr_acc']:.3f} | "
                     f"VaL {val_metrics['loss']:.3f} VaAcc {val_metrics['acc']:.3f} VaFR {val_metrics['fr_acc']:.3f}")

    # 4. Final Evaluation at specified erasure rates
    final_results = []
    for e_rate in cfg.erasure_rates:
        eval_metrics = run_epoch(model, dl_test, None, None, device, cfg, is_train=False, eval_er=e_rate)
        result = {
            "backbone": cfg.backbone, "dataset": cfg.dataset,
            "dropout_p": cfg.dp, "noise_std": cfg.noise, "erasure": e_rate,
            "accuracy": eval_metrics["acc"], "f1": eval_metrics["f1"], "fr_accuracy": eval_metrics["fr_acc"],
            "redundancy": cfg.redundancy, "learnable_A": cfg.learnable_A,
            "aux_rec_loss": cfg.aux_rec_loss, "lambda_rec": cfg.lambda_rec,
            "no_nncl": cfg.no_nncl, "final_eval_loss": eval_metrics["loss"],
            "seed": cfg.seed, "epochs": cfg.epochs, "batch_size": cfg.batch,
            "warmup_epochs": cfg.warmup_epochs
        }
        final_results.append(result)
        logging.info(f"[Final Eval] Erasure {e_rate:.2f} | Acc {eval_metrics['acc']:.4f} F1 {eval_metrics['f1']:.4f} FR {eval_metrics['fr_acc']:.4f}")

    return final_results

# --- CLI and Main Execution ---

def cli():
    """Command-line interface setup."""
    p = argparse.ArgumentParser(description="NNCL Experiment Runner")
    # ... (rest of the CLI arguments are fine, no changes needed here)
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10", help="Dataset to use.")
    p.add_argument("--data_root", type=Path, default=Path("./data"), help="Root directory for datasets.")
    p.add_argument("--backbones", nargs="+", default=["resnet18"], help="List of backbone models to run.")
    p.add_argument("--redundancy", type=float, default=0.25, help="NNCL redundancy factor k.")
    p.add_argument("--learnable_A", action="store_true", help="Set the NNCL coding matrix A to be learnable.")
    p.add_argument("--no_nncl", action="store_true", help="Run baseline model without NNCL.")
    p.add_argument("--dropout_rates", nargs="+", type=float, default=[0.0], help="List of dropout probabilities for training.")
    p.add_argument("--noise_stds", nargs="+", type=float, default=[0.0], help="List of Gaussian noise std deviations for input.")
    p.add_argument("--erasure_rates", nargs="+", type=float, default=[0.0, 0.3, 0.6], help="List of feature erasure rates for evaluation.")
    p.add_argument("--epochs", type=int, default=30, help="Total number of training epochs.")
    p.add_argument("--batch", type=int, default=128, help="Batch size.")
    p.add_argument("--warmup_epochs", type=int, default=5, help="Number of linear warmup epochs.")
    p.add_argument("--aux_rec_loss", action="store_true", help="Enable auxiliary reconstruction loss.")
    p.add_argument("--lambda_rec", type=float, default=1.0, help="Weight for the auxiliary reconstruction loss.")
    p.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic execution for reproducibility.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu).")
    p.add_argument("--csv_out", type=Path, default=Path("results/all_experiments_summary.csv"), help="Path to save the summary CSV file.")
    p.add_argument("--use_masked_var", action="store_true", help="Use variance of masked elements for FR metric (default: full variance).")
    p.add_argument("--lr", type=float, default=1e-4, help="Base learning rate.")
    return p.parse_args()

def main():
    """Main function to run experiments based on CLI arguments."""
    args = cli()
    logging.info(f"CLI arguments: {vars(args)}")
    device = torch.device(args.device)

    all_results = []
    dp = args.dropout_rates[0]
    noise = args.noise_stds[0]

    for bb in args.backbones:
        cfg = argparse.Namespace(**vars(args))
        cfg.backbone = bb
        cfg.dp = dp
        cfg.noise = noise
        
        logging.info(f"\n{'='*20} Starting Experiment {'='*20}")
        logging.info(f"Backbone: {bb}, Dataset: {cfg.dataset}, Dropout: {dp}, Noise: {noise}, Seed: {cfg.seed}")
        
        current_run_results = run_single_experiment(cfg, device)
        all_results.extend(current_run_results)
        
        logging.info(f"{'='*22} Finished Experiment {'='*22}\n")

    if all_results:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
            logging.info(f"All experiment results saved to -> {args.csv_out}")
        except IOError as e:
            logging.error(f"Failed to write results to {args.csv_out}: {e}")
    else:
        logging.warning("No experiments were run, so no results were saved.")

if __name__ == "__main__":
    main()
