# train.py
"""Oxford-IIIT Pet 细粒度分类训练主入口。

用法示例：
    python train.py --out_dir ./runs/baseline
    python train.py --out_dir ./runs/ls01 --label_smoothing 0.1
    python train.py --out_dir ./runs/randaug --randaugment
    python train.py --out_dir ./runs/frozen --freeze_backbone

三个消融开关（每次实验只改其中一个，保证单变量）：
    --randaugment        数据增强：基础增强 -> 进阶增强
    --freeze_backbone    微调策略：全参数微调 -> 仅微调 fc
    --label_smoothing    损失函数：CrossEntropy -> Label Smoothing
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import NUM_CLASSES, build_dataloaders
from models.model import build_model, count_parameters


# ---------- 命令行参数 ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet 训练脚本")
    # 数据与输出
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./runs/baseline",
                        help="日志与权重的输出目录，每次实验用不同目录")
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9, help="仅 SGD 使用")
    # 消融开关
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="0 表示不用；消融实验用 0.1")
    parser.add_argument("--mixup", type=float, default=0.0,
                        help="0 表示不用；常用 0.2（Beta 分布参数）")
    parser.add_argument("--randaugment", action="store_true",
                        help="训练集叠加 RandAugment 进阶增强")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="冻结骨干，仅训练新的分类头")
    # 运行环境
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练（提速）")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- MixUp ----------
def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """对 batch 做 MixUp：把两张图按比例线性融合，标签也按同比例混合。

    返回融合后的输入、两组标签、以及混合系数 lam。
    """
    lam = float(np.random.beta(alpha, alpha))
    # 固定 lam >= 0.5，保证主标签是 y_a，避免训练目标来回摇摆
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    return x_mixed, y, y[perm], lam


# ---------- 单轮训练 ----------
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    mixup_alpha: float,
) -> float:
    model.train()
    running_loss = 0.0
    n_seen = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # 梯度清零必须放在 backward 之前：PyTorch 默认会累积梯度
        optimizer.zero_grad(set_to_none=True)

        if mixup_alpha > 0.0:
            x, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
        else:
            y_a = y_b = y
            lam = 1.0

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            if mixup_alpha > 0.0:
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, y_a)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        n_seen += x.size(0)
        pbar.set_postfix(loss=f"{running_loss / n_seen:.4f}")

    return running_loss / max(n_seen, 1)


# ---------- 验证 ----------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """在给定数据集上评估，返回 (loss, top1_acc, macro_f1)。

    注意：必须 model.eval() + no_grad()，否则 BatchNorm 会更新统计量、
    且会白白构建计算图浪费显存。
    """
    model.eval()
    running_loss = 0.0
    n_seen = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * y.size(0)
        n_seen += y.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    acc = 100.0 * sum(p == t for p, t in zip(all_preds, all_labels)) / max(n_seen, 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return running_loss / max(n_seen, 1), acc, macro_f1


# ---------- 主流程 ----------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # cudnn 自动选最快卷积算法（输入尺寸固定时收益明显）
    torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print(f"设备      : {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"输出目录  : {out_dir}")
    print("=" * 60)

    # 1. 数据
    train_loader, val_loader, _ = build_dataloaders(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        rand_augment=args.randaugment,
    )
    print(f"训练集 {len(train_loader.dataset)} | 验证集 {len(val_loader.dataset)}")

    # 2. 模型
    model = build_model(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    trainable, total = count_parameters(model)
    print(f"可训练参数 {trainable:,} / 总参数 {total:,}")

    # 3. 损失函数（label_smoothing>0 即为消融实验组）
    #    注意 CrossEntropyLoss 自带 Softmax，模型输出层不能再加 Softmax
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 4. 优化器：只传入 requires_grad=True 的参数（冻结骨干时跳过那些层）
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )

    # 5. 混合精度（关闭时 GradScaler 是空操作）
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    use_amp = args.amp and device.type == "cuda"

    writer = SummaryWriter(log_dir=str(out_dir))
    writer.add_text("config", json.dumps(vars(args), indent=2, ensure_ascii=False))

    # 6. 训练循环
    best_acc = 0.0
    best_path = out_dir / "best_model.pth"
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, args.mixup
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        # TensorBoard 记录
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val_top1", val_acc, epoch)
        writer.add_scalar("F1/val_macro", val_f1, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        flag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "args": vars(args),
                },
                best_path,
            )
            flag = "  <- best"

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"val_acc {val_acc:.2f}% | macro_f1 {val_f1:.4f} | {dt:.1f}s{flag}"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
             "val_acc": val_acc, "val_f1": val_f1, "seconds": dt}
        )

    writer.close()
    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    print("=" * 60)
    print(f"训练结束 | 最佳验证集 Top-1 准确率: {best_acc:.2f}%")
    print(f"最佳权重: {best_path}")
    print(f"TensorBoard 日志: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # Windows 上 num_workers>0 必须有这个保护，否则子进程会无限重启
    main()
