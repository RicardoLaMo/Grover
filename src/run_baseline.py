"""
End-to-End baseline training script for IEEE-CIS and SoFi datasets.

Runs the TabTransformer baseline with the paper's default hyperparameters,
saves a sample of the training data for audit, evaluates with ROC-AUC and
PR-AUC, and generates results ready for the LaTeX experiment reporter.

Usage:
  python src/run_baseline.py --dataset ieee --sample-size 50000 --max-epochs 20
  python src/run_baseline.py --dataset sofi --max-epochs 20
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tab_transformer import TabTransformer
from src.data.data_loader import (
    load_ieee_cis,
    load_sample,
    get_column_info,
    prepare_data,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run TabTransformer baseline")
    parser.add_argument("--dataset", choices=["ieee", "sofi"], default="ieee")
    parser.add_argument("--sample-size", type=int, default=50000,
                        help="Sample size from full data (for IEEE)")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="output/phase1_baseline")
    parser.add_argument("--col-embed-mode", choices=["concat", "add"], default="concat")
    parser.add_argument("--audit-sample", type=int, default=1000,
                        help="Rows to save from training data for audit")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_batches = 0, 0
    for batch in loader:
        x_cat = batch["x_cat"].to(device)
        x_cont = batch["x_cont"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        out = model(x_cat, x_cont)
        loss = criterion(out["logits"].squeeze(-1), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0, 0
    all_probs, all_labels = [], []
    for batch in loader:
        x_cat = batch["x_cat"].to(device)
        x_cont = batch["x_cont"].to(device)
        y = batch["y"].to(device)
        out = model(x_cat, x_cont)
        logits = out["logits"].squeeze(-1)
        loss = criterion(logits, y)
        total_loss += loss.item()
        n_batches += 1
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / max(n_batches, 1)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    return avg_loss, roc_auc, pr_auc


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    # ---------- Load data ----------
    if args.dataset == "ieee":
        df = load_ieee_cis("data/IEEE", args.sample_size)
    else:
        df = load_sample("sofi_small", data_dir="data/samples")

    col_info = get_column_info(df)
    print(f"Categorical: {len(col_info['categorical'])}, Continuous: {len(col_info['continuous'])}")

    # ---------- Save audit sample ----------
    os.makedirs(args.output_dir, exist_ok=True)
    audit_path = os.path.join(args.output_dir, f"audit_sample_{args.dataset}.csv")
    audit_n = min(args.audit_sample, len(df))
    df.sample(n=audit_n, random_state=42).to_csv(audit_path, index=False)
    print(f"Audit sample saved: {audit_path} ({audit_n} rows)")

    # ---------- Prepare data ----------
    data = prepare_data(df, col_info)
    train_ds, val_ds, test_ds = data["train"], data["val"], data["test"]
    encoder = data["encoder"]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2)

    # ---------- Build model ----------
    model = TabTransformer(
        num_categories_per_col=encoder.num_categories_per_col,
        num_continuous=len(col_info["continuous"]),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        col_embed_mode=args.col_embed_mode,
    ).to(device)
    print(f"Model params: {model.num_parameters:,}")

    # ---------- Loss & optimizer ----------
    y_train = train_ds.y.numpy()
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    # ---------- Training loop ----------
    best_val_auc = 0
    patience_counter = 0
    t_start = time.time()

    print(f"\n{'Ep':>3} {'TrLoss':>8} {'VLoss':>8} {'V-AUC':>7} {'V-PR':>7} {'LR':>9} {'t':>5}")
    print("-" * 55)

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, val_pr_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(f"{epoch:3d} {train_loss:8.4f} {val_loss:8.4f} {val_auc:7.4f} {val_pr_auc:7.4f} {lr:9.2e} {elapsed:5.1f}s")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    total_time = time.time() - t_start

    # ---------- Final test ----------
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"), weights_only=True))
    test_loss, test_auc, test_pr_auc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*50}")
    print(f"DATASET: {args.dataset.upper()}")
    print(f"{'='*50}")
    print(f"  ROC-AUC:  {test_auc:.4f}")
    print(f"  PR-AUC:   {test_pr_auc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Training time: {total_time:.1f}s")
    print(f"{'='*50}")

    # ---------- Log experiment ----------
    log_entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "category": f"TabTransformer-Baseline-{args.dataset.upper()}",
        "parent_idea": None,
        "prd_link": "docs/prd/drill_baseline.md",
        "git_branch": "main",
        "idea": f"TabTransformer baseline on {args.dataset.upper()} data with concat column embedding",
        "parameters": {
            "dataset": args.dataset,
            "sample_size": args.sample_size if args.dataset == "ieee" else "full",
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "col_embed_mode": args.col_embed_mode,
            "n_cat_features": len(col_info["categorical"]),
            "n_cont_features": len(col_info["continuous"]),
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "fraud_rate": float(df["isFraud"].mean()),
        },
        "results": {
            "test_roc_auc": round(test_auc, 4),
            "test_pr_auc": round(test_pr_auc, 4),
            "test_loss": round(test_loss, 4),
            "best_val_roc_auc": round(best_val_auc, 4),
            "num_params": model.num_parameters,
            "training_time_s": round(total_time, 1),
        },
        "is_champion": True,
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/experiment_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Save results as standalone JSON for LaTeX reporter
    results_path = os.path.join(args.output_dir, f"results_{args.dataset}.json")
    with open(results_path, "w") as f:
        json.dump(log_entry, f, indent=2)

    print(f"\nExperiment logged to artifacts/experiment_log.jsonl")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
