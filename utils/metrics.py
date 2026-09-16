# utils/metrics.py
"""评估指标计算与混淆矩阵可视化。

包含：
- Top-1 / Top-5 准确率
- Macro-F1（细粒度分类比 Acc 更能暴露「某些品种完全学不会」的问题）
- 37 类混淆矩阵绘制（原始计数 + 行归一化两个版本）
- 混淆最严重的类别对搜索
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # 无 GUI 后端：脚本运行时不弹窗、不需要显示器
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, f1_score

# 让图里的中文正常显示（否则会变成方框）；Windows 自带微软雅黑/黑体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 负号不显示成豆腐块


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> List[float]:
    """计算 Top-k 准确率（百分比）。

    Args:
        logits: [N, C] 模型原始输出（未过 Softmax）。
        targets: [N] 真实标签。
        topk: 要计算的 k 值。

    Returns:
        与 topk 一一对应的准确率列表。
    """
    n = targets.size(0)
    maxk = min(max(topk), logits.size(1))

    # topk 返回 (values, indices)，按分数从高到低排序
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # [maxk, N]
    # 逐位置比较是否等于真实标签 -> [maxk, N] 的布尔矩阵
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results: List[float] = []
    for k in topk:
        # any(dim=0)：真实标签只要落在前 k 个里就算对
        correct_k = correct[:k].any(dim=0).sum().item()
        results.append(100.0 * correct_k / n)
    return results


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, object]:
    """汇总 Top-1 / Top-5 / Macro-F1 与混淆矩阵。"""
    top1, top5 = topk_accuracy(logits, targets, topk=(1, 5))
    preds = logits.argmax(dim=1).numpy()
    labels = targets.numpy()

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    # labels=range(...) 保证即使某个类一次都没被预测到，矩阵里也有它的行
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    return {"top1": top1, "top5": top5, "macro_f1": macro_f1, "cm": cm,
            "preds": preds, "labels": labels}


def find_top_confused_pairs(
    cm: np.ndarray,
    class_names: Sequence[str],
    top_n: int = 5,
) -> List[Tuple[int, str, str, int, int]]:
    """找出混淆最严重的类别对（对称统计，排除对角线）。

    Returns:
        [(合计错分次数, 类别A, 类别B, A->B 次数, B->A 次数), ...]，按严重程度降序。
    """
    off = cm.copy()
    np.fill_diagonal(off, 0)  # 对角线是「预测正确」，不算混淆

    pairs: List[Tuple[int, str, str, int, int]] = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):  # 只取上三角，避免重复
            total = int(off[i, j] + off[j, i])
            if total > 0:
                pairs.append((total, class_names[i], class_names[j],
                              int(off[i, j]), int(off[j, i])))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:top_n]


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: Path | str,
    normalize: bool = False,
) -> None:
    """绘制 37 类混淆矩阵热力图。

    Args:
        normalize: True 时按行归一化（每行除以该类的真实样本数），
                   更能看出「某个品种被错分到哪去了」的比例。
    """
    if normalize:
        matrix = cm.astype(np.float64)
        row_sum = matrix.sum(axis=1, keepdims=True)
        # 防止除零：没有样本的类别保持全 0
        matrix = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
        fmt = ".2f"
        title = "Confusion Matrix (row-normalized)"
    else:
        # 保持整数类型，否则 fmt="d" 无法格式化浮点数
        matrix = cm
        fmt = "d"
        title = "Confusion Matrix (counts)"

    plt.figure(figsize=(22, 20))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 6},
        cbar_kws={"shrink": 0.7},
    )
    plt.title(title, fontsize=18)
    plt.xlabel("Predicted label", fontsize=14)
    plt.ylabel("True label", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
