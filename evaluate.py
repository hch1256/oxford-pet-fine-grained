# evaluate.py
"""加载训练好的权重，在测试集上评估并生成可视化结果。

产出（全部写入权重所在目录）：
    metrics.json               Top-1 / Top-5 / Macro-F1
    confusion_matrix.png       37 类混淆矩阵（计数）
    confusion_matrix_norm.png  37 类混淆矩阵（行归一化）
    gradcam_correct.png        预测正确样本的 Grad-CAM
    gradcam_error.png          预测错误样本的 Grad-CAM
    gradcam_compare.png        两者并排对比图
    misclassified.txt          全部误判样本清单

用法：
    python evaluate.py --ckpt ./runs/baseline/best_model.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# 中文字体（否则图内中文标题会渲染成方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data.dataset import NUM_CLASSES, build_dataloaders, build_transforms, get_class_names
from models.model import build_model, get_gradcam_target_layer
from utils.gradcam import generate_gradcam
from utils.metrics import compute_metrics, find_top_confused_pairs, plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet 评估与可视化")
    parser.add_argument("--ckpt", type=str, default="./runs/baseline/best_model.pth",
                        help="训练好的权重路径")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="结果输出目录，默认与权重同目录")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_confused", type=int, default=5,
                        help="打印混淆最严重的类别对数量")
    return parser.parse_args()


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """跑一遍数据集，收集全部 logits 与真实标签（顺序与 dataset 一致）。"""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for x, y in tqdm(loader, desc="evaluate", leave=False):
        x = x.to(device, non_blocking=True)
        all_logits.append(model(x).cpu())
        all_labels.append(y)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def pick_gradcam_samples(
    labels, preds, confs, cm, class_names
):
    """挑选两个用于 Grad-CAM 的典型样本。

    1. 预测正确的样本：取置信度最高的那个（模型最「自信且正确」）
    2. 预测错误的样本：从混淆最严重的类别对里挑置信度最高的（错得最「自信」）
    """
    correct_idx = [i for i in range(len(labels)) if labels[i] == preds[i]]
    best_correct = max(correct_idx, key=lambda i: confs[i]) if correct_idx else None

    pairs = find_top_confused_pairs(cm, class_names, top_n=1)
    error_idx = [i for i in range(len(labels)) if labels[i] != preds[i]]
    if pairs:
        _, name_a, name_b, _, _ = pairs[0]
        idx_a, idx_b = class_names.index(name_a), class_names.index(name_b)
        # 只保留「这一对之间互相混淆」的误判样本
        confusable = [
            i for i in error_idx
            if {labels[i], preds[i]} == {idx_a, idx_b}
        ]
        pool = confusable if confusable else error_idx
    else:
        pool = error_idx
    best_error = max(pool, key=lambda i: confs[i]) if pool else None

    return best_correct, best_error


def make_gradcam_figure(
    info: dict, save_path: Path, row_titles: list[str]
) -> None:
    """把 2 个样本的「原图 | 热力图」拼成一张 2x2 对比图。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for row, (item, row_title) in enumerate(zip(info, row_titles)):
        axes[row][0].imshow(item["rgb"])
        axes[row][0].set_title(f"{row_title}\n原图", fontsize=12)
        axes[row][0].axis("off")

        axes[row][1].imshow(item["overlay"])
        axes[row][1].set_title(
            f"真实: {item['true_name']}\n预测: {item['pred_name']} ({item['conf']:.1%})",
            fontsize=12,
        )
        axes[row][1].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_single_gradcam(info: dict, save_path: Path, title: str) -> None:
    """单独保存一张「原图 | 热力图」并排图。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(info["rgb"])
    axes[0].set_title("原图", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(info["overlay"])
    axes[1].set_title(
        f"真实: {info['true_name']} | 预测: {info['pred_name']} ({info['conf']:.1%})",
        fontsize=11,
    )
    axes[1].axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载权重（pretrained=False：加载自己的权重，无需再下 ImageNet 权重）
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    train_args = ckpt.get("args", {})
    print(f"已加载权重: {ckpt_path}")
    print(f"  训练轮数: {ckpt.get('epoch')}  最佳验证准确率: {ckpt.get('best_acc'):.2f}%")
    print(f"  训练配置: randaugment={train_args.get('randaugment')}, "
          f"label_smoothing={train_args.get('label_smoothing')}, "
          f"freeze_backbone={train_args.get('freeze_backbone')}, "
          f"mixup={train_args.get('mixup')}")

    # 2. 测试集
    _, _, test_loader = build_dataloaders(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    samples = test_loader.dataset.samples
    class_names = get_class_names(args.data_dir)
    print(f"测试集样本数: {len(samples)}  类别数: {len(class_names)}")

    # 3. 指标
    logits, labels = collect_logits(model, test_loader, device)
    metrics = compute_metrics(logits, labels, NUM_CLASSES)
    preds = metrics["preds"]
    cm = metrics["cm"]

    probs = torch.softmax(logits, dim=1)
    confs = probs.max(dim=1).values.numpy()

    print("\n" + "=" * 56)
    print(f"Top-1 Accuracy : {metrics['top1']:.2f}%")
    print(f"Top-5 Accuracy : {metrics['top5']:.2f}%")
    print(f"Macro-F1       : {metrics['macro_f1']:.4f}")
    print("=" * 56)

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "ckpt": str(ckpt_path),
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "macro_f1": metrics["macro_f1"],
                "train_args": train_args,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 4. 混淆矩阵（计数 + 归一化两个版本）
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix_norm.png", normalize=True)
    print(f"\n混淆矩阵已保存: {out_dir / 'confusion_matrix.png'}")

    # 5. 最容易混淆的类别对（报告素材）
    pairs = find_top_confused_pairs(cm, class_names, top_n=args.top_confused)
    print("\n最容易混淆的类别对（合计错分次数 | A->B | B->A）:")
    for total, name_a, name_b, ab, ba in pairs:
        print(f"  {total:>4}  {name_a:<26} <-> {name_b:<26}  ({ab} | {ba})")

    # 6. 误判样本清单（写进报告）
    lines = ["真实标签\t预测标签\t置信度\t文件路径"]
    for i in range(len(labels)):
        if labels[i] != preds[i]:
            lines.append(
                f"{class_names[labels[i]]}\t{class_names[preds[i]]}\t"
                f"{confs[i]:.4f}\t{Path(samples[i][0]).name}"
            )
    (out_dir / "misclassified.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n误判样本清单已保存: {out_dir / 'misclassified.txt'} "
          f"（共 {len(lines) - 1} 个）")

    # 7. Grad-CAM：一个预测正确、一个预测错误
    eval_tf = build_transforms()[1]
    target_layer = get_gradcam_target_layer(model)

    idx_correct, idx_error = pick_gradcam_samples(labels, preds, confs, cm, class_names)
    selected = []
    for tag, idx in (("correct", idx_correct), ("error", idx_error)):
        if idx is None:
            print(f"没有可用于 Grad-CAM 的 {tag} 样本")
            continue
        info = generate_gradcam(
            model=model,
            image_path=samples[idx][0],
            transform=eval_tf,
            target_layer=target_layer,
            device=device,
        )
        info["true_name"] = class_names[labels[idx]]
        info["pred_name"] = class_names[preds[idx]]
        info["file"] = Path(samples[idx][0]).name
        selected.append((tag, info))
        save_single_gradcam(
            info,
            out_dir / f"gradcam_{tag}.png",
            f"{'预测正确' if tag == 'correct' else '预测错误'} · {info['file']}",
        )
        print(f"\nGrad-CAM ({tag}): {info['file']}")
        print(f"  真实 {info['true_name']} | 预测 {info['pred_name']} ({info['conf']:.1%})")

    if len(selected) == 2:
        make_gradcam_figure(
            [info for _, info in selected],
            out_dir / "gradcam_compare.png",
            ["预测正确（置信度最高）", "预测错误（最易混类别对）"],
        )
        print(f"\nGrad-CAM 对比图已保存: {out_dir / 'gradcam_compare.png'}")

    print(f"\n全部结果输出目录: {out_dir}")


if __name__ == "__main__":
    main()
