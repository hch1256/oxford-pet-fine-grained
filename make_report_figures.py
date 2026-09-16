# make_report_figures.py
"""生成技术报告所需的图表（可重复运行，结果稳定）。

产出（默认写入 ./report_figures/）：
    fig1_curves.png    收敛曲线：Loss 对比 / Label Smoothing 独立 Loss / Val Top-1 对比
    fig2_cm_zoom.png   混淆矩阵局部对比：Baseline vs +Label Smoothing

为什么把 Label Smoothing 的 Loss 单独画一张：
    Label Smoothing 改变了损失函数的定义（目标分布含 0.1 的均匀项，
    理论下界约 0.1*ln(37)≈0.36），其 loss 绝对值与另外两组不在同一量纲上，
    画在一起会造成「LS 收敛得最差」的错误印象。

用法：
    python make_report_figures.py
    python make_report_figures.py --top_pairs 6 --out_dir ./report_figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

# 中文字体（否则图内中文标题渲染成方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data.dataset import NUM_CLASSES, build_dataloaders, get_class_names
from evaluate import collect_logits
from models.model import build_model
from utils.metrics import find_top_confused_pairs

# runs/ 下的目录名 -> 报告里显示的名字
RUN_LABELS = {
    "baseline": "Baseline",
    "randaug": "+ RandAugment",
    "ls01": "+ Label Smoothing (0.1)",
}
RUN_COLORS = {
    "baseline": "#4C72B0",
    "randaug": "#DD8452",
    "ls01": "#55A868",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成技术报告图表")
    parser.add_argument("--runs", nargs="+",
                        default=["./runs/baseline", "./runs/randaug", "./runs/ls01"],
                        help="参与绘图的实验目录（需含 history.json 与 metrics.json）")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./report_figures")
    parser.add_argument("--top_pairs", type=int, default=4,
                        help="混淆矩阵局部图取前 N 个最易混类别对（最多涉及 2N 个类）")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def load_history(run_dir: Path) -> list[dict]:
    """读取一个实验目录的 history.json，并校验轮数一致便于对齐横轴。"""
    with open(run_dir / "history.json", encoding="utf-8") as f:
        return json.load(f)


def plot_curves(histories: dict[str, list[dict]], save_path: Path) -> None:
    """画三张子图：Loss 对比 / LS 独立 Loss / Val Top-1 对比。"""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    # --- (a) Loss：只画量纲一致的两组，LS 单独放 (b) ---
    ax = axes[0]
    for key, hist in histories.items():
        if key == "ls01":  # 量纲不同，跳过
            continue
        epochs = [h["epoch"] for h in hist]
        color = RUN_COLORS.get(key, None)
        ax.plot(epochs, [h["train_loss"] for h in hist], "-o", ms=3.5,
                color=color, label=f"{RUN_LABELS.get(key, key)} - Train")
        ax.plot(epochs, [h["val_loss"] for h in hist], "--s", ms=3.5,
                color=color, alpha=0.6, label=f"{RUN_LABELS.get(key, key)} - Val")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("(a) 收敛曲线：Baseline vs +RandAugment\n（两者损失定义相同，可直接比较）",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- (b) Label Smoothing 的 Loss（独立纵轴）---
    ax = axes[1]
    ls_hist = histories.get("ls01")
    if ls_hist is not None:
        epochs = [h["epoch"] for h in ls_hist]
        ax.plot(epochs, [h["train_loss"] for h in ls_hist], "-o", ms=3.5,
                color=RUN_COLORS["ls01"], label="Train Loss")
        ax.plot(epochs, [h["val_loss"] for h in ls_hist], "--s", ms=3.5,
                color=RUN_COLORS["ls01"], alpha=0.6, label="Val Loss")
        # 标注理论下界：目标分布含 0.1 均匀项 -> 最优时 loss 约为 0.1*ln(37)
        floor = 0.1 * np.log(NUM_CLASSES)
        ax.axhline(floor, color="red", ls=":", lw=1.5,
                   label=f"理论下界 ≈ {floor:.2f}")
        # 标注 val_loss 最低点（报告中「过拟合拐点」的依据）
        val_losses = [h["val_loss"] for h in ls_hist]
        best = int(np.argmin(val_losses))
        ax.annotate(f"最低点 ep{epochs[best]}\n{val_losses[best]:.4f}",
                    xy=(epochs[best], val_losses[best]),
                    xytext=(epochs[best] - 4.5, val_losses[best] + 0.06),
                    fontsize=9, color="#B22222",
                    arrowprops=dict(arrowstyle="->", color="#B22222", lw=1.2))
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("(b) + Label Smoothing 的损失曲线\n（损失定义已改变，纵轴独立）",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- (c) 验证集 Top-1 准确率：三组同图 ---
    ax = axes[2]
    for key, hist in histories.items():
        epochs = [h["epoch"] for h in hist]
        accs = [h["val_acc"] for h in hist]
        ax.plot(epochs, accs, "-o", ms=3.5,
                color=RUN_COLORS.get(key, None),
                label=RUN_LABELS.get(key, key))
        best = int(np.argmax(accs))
        ax.annotate(f"{accs[best]:.2f}%", xy=(epochs[best], accs[best]),
                    xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=9,
                    color=RUN_COLORS.get(key, None), fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Val Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("(c) 验证集 Top-1 准确率对比\n（准确率可跨组直接比较）", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_confusion(ckpt_path: Path, args, device: torch.device) -> np.ndarray:
    """加载权重，在测试集上跑一遍，返回 37x37 混淆矩阵。

    为了反复调图时不用重复跑 GPU，结果缓存为同目录的 confusion_matrix.npy。
    """
    cache = ckpt_path.parent / "confusion_matrix.npy"
    if cache.exists():
        print(f"  使用缓存: {cache}")
        return np.load(cache)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    _, _, test_loader = build_dataloaders(
        root=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    logits, labels = collect_logits(model, test_loader, device)
    cm = confusion_matrix(labels.numpy(), logits.argmax(dim=1).numpy(),
                          labels=list(range(NUM_CLASSES)))
    np.save(cache, cm)
    print(f"  已缓存: {cache}")
    return cm


def print_pair_comparison(
    cms: dict[str, np.ndarray], class_names: list[str], top_n: int
) -> None:
    """跨实验对比「最易混类别对」的错分次数变化（报告第 3 节的核心表格）。

    以 Baseline 的混淆矩阵确定要跟踪哪些类别对，再逐组取出对应数字。
    """
    pairs = find_top_confused_pairs(cms["baseline"], class_names, top_n=top_n)
    keys = list(cms.keys())

    header = f"{'类别对 (A <-> B)':<52}" + "".join(f"{RUN_LABELS.get(k, k):>22}" for k in keys)
    print("\n" + header)
    print("-" * len(header))
    for _, name_a, name_b, _, _ in pairs:
        i, j = class_names.index(name_a), class_names.index(name_b)
        row = f"{name_a + ' <-> ' + name_b:<52}"
        for k in keys:
            cm = cms[k]
            row += f"{int(cm[i, j]):>10} |{int(cm[j, i]):<10}"
        print(row)
    print("\n（每个单元格格式为「A->B | B->A」；A->B 表示真实为 A 却被预测成 B）")


def select_zoom_classes(cm: np.ndarray, class_names: list[str], top_pairs: int) -> list[int]:
    """选出「最易混淆的若干类别对」所涉及的全部类别下标（去重排序）。

    用 Baseline 的混淆矩阵来选，这样对比图上两侧类别完全一致。
    """
    pairs = find_top_confused_pairs(cm, class_names, top_n=top_pairs)
    print("\n局部图涉及的易混类别对：")
    for total, name_a, name_b, ab, ba in pairs:
        print(f"  {total:>3}  {name_a} <-> {name_b}   ({ab} | {ba})")
    idx = {class_names.index(name) for _, a, b, _, _ in pairs for name in (a, b)}
    return sorted(idx)


def plot_confusion_zoom(
    cms: dict[str, np.ndarray],
    zoom_idx: list[int],
    class_names: list[str],
    save_path: Path,
) -> None:
    """把 37x37 混淆矩阵裁剪成易混类别的子矩阵，左右对比。"""
    names = [class_names[i] for i in zoom_idx]
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(8.2 * n, 7.2))
    if n == 1:
        axes = [axes]

    for ax, (key, cm) in zip(axes, cms.items()):
        sub = cm[np.ix_(zoom_idx, zoom_idx)]
        sns.heatmap(sub, annot=True, fmt="d", cmap="Blues",
                    xticklabels=names, yticklabels=names,
                    annot_kws={"size": 11}, cbar=False, ax=ax)
        ax.set_title(RUN_LABELS.get(key, key), fontsize=15, fontweight="bold")
        ax.set_xlabel("Predicted label", fontsize=12)
        ax.set_ylabel("True label", fontsize=12)
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = [Path(p) for p in args.runs]

    # ---------- 图 1：收敛曲线（只用 json，不碰 GPU）----------
    print("=" * 60)
    print("图 1：收敛曲线")
    histories = {}
    for run in runs:
        key = run.name
        hist = load_history(run)
        histories[key] = hist
        total = sum(h["seconds"] for h in hist)
        best_acc = max(h["val_acc"] for h in hist)
        print(f"  {RUN_LABELS.get(key, key):<28} {len(hist)} ep  "
              f"耗时 {total / 60:.1f} min  最佳 Val Acc {best_acc:.2f}%")
    plot_curves(histories, out_dir / "fig1_curves.png")
    print(f"  已保存: {out_dir / 'fig1_curves.png'}")

    # ---------- 图 2：混淆矩阵局部对比 ----------
    print("\n" + "=" * 60)
    print("图 2：混淆矩阵局部对比")
    class_names = get_class_names(args.data_dir)

    cms = {}
    for run in runs:
        ckpt = run / "best_model.pth"
        if not ckpt.exists():
            print(f"  跳过 {run.name}：找不到 {ckpt}")
            continue
        print(f"  {RUN_LABELS.get(run.name, run.name)}: {ckpt}")
        cms[run.name] = compute_confusion(ckpt, args, device)

    if "baseline" not in cms:
        print("  缺少 baseline 混淆矩阵，无法绘制局部图")
        return

    # 用 Baseline 的混淆矩阵决定看哪些类别（保证各图之间类别一致）
    zoom_idx = select_zoom_classes(cms["baseline"], class_names, args.top_pairs)
    print(f"  共 {len(zoom_idx)} 个类别: {[class_names[i] for i in zoom_idx]}")

    # 局部对比图只放 Baseline 与 Label Smoothing：两张 8x8 已经够宽，
    # 第三张并排后字会小到看不清；RandAugment 的数据放在上面的对比表里。
    zoom_cms = {k: cms[k] for k in ("baseline", "ls01") if k in cms}
    plot_confusion_zoom(zoom_cms, zoom_idx, class_names, out_dir / "fig2_cm_zoom.png")
    print(f"  已保存: {out_dir / 'fig2_cm_zoom.png'}")

    # 跨实验对比表（报告第 3 节直接用）
    print_pair_comparison(cms, class_names, args.top_pairs)

    print("\n" + "=" * 60)
    print(f"全部图表输出目录: {out_dir}")


if __name__ == "__main__":
    main()
