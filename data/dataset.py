# data/dataset.py
"""Oxford-IIIT Pet 数据加载、70:15:15 分层划分与 Transform。

设计要点：
1. samples 里只存 (图片路径, 标签)，**不预先加载图片**。图片在 __getitem__ 里
   按需打开——这样 num_workers 多进程时，dataset 被 pickle 给子进程的开销
   只有几 MB；若预先把图片全读进内存，多进程会复制好几份，直接把内存撑爆。
2. 训练集与验证/测试集使用**两套不同的 transform**，因此拆成独立的
   PetDataset 实例，而不是共用一个底层 dataset 再套 Subset。
"""

from __future__ import annotations

import pathlib
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

# 预训练 ResNet 在 ImageNet 上训练时使用的归一化统计量，必须保持一致
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NUM_CLASSES = 37


# ---------- 1. 自定义 Dataset 包装类 ----------
class PetDataset(Dataset):
    """持有一份 (图片路径, 标签) 列表和一个 transform。

    之所以不用 Subset：train / val 需要不同的 transform，而 Subset 会共享
    底层 dataset 的 transform，无法分别指定。
    """

    def __init__(
        self,
        samples: List[Tuple[pathlib.Path, int]],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        # convert("RGB") 不可省：数据集含灰度图，不转 3 通道会与模型输入不匹配
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ---------- 2. Transform ----------
def build_transforms(rand_augment: bool = False) -> Tuple[Callable, Callable]:
    """构建训练集与评估集的 Transform。

    Args:
        rand_augment: 是否在训练集上叠加 RandAugment 进阶增强（消融实验用）。
    """
    train_ops: List[Callable] = [
        transforms.Resize(256),          # 短边缩到 256，给裁剪留余量
        transforms.RandomCrop(224),      # 数据增强：随机位置裁剪
        transforms.RandomHorizontalFlip(),  # 数据增强：随机水平翻转（仅训练集！）
    ]
    if rand_augment:
        # RandAugment 会随机组合亮度/对比度/旋转/平移等操作，必须在 ToTensor 之前
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_ops += [
        transforms.ToTensor(),           # PIL -> Tensor，[0,255]->[0,1]，HWC->CHW
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    train_tf = transforms.Compose(train_ops)

    # 验证/测试集必须保持确定性：不可加入 RandomCrop / RandomHorizontalFlip
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


# ---------- 3. 分层划分 ----------
def stratified_split(
    root: str = "./data",
    seed: int = 42,
) -> Tuple[
    List[Tuple[pathlib.Path, int]],
    List[Tuple[pathlib.Path, int]],
    List[Tuple[pathlib.Path, int]],
]:
    """合并 trainval + test 成全量，再按 70:15:15 分层划分为 train/val/test。"""
    ds_trainval = OxfordIIITPet(
        root=root, split="trainval", target_types="category", download=True
    )
    ds_test = OxfordIIITPet(
        root=root, split="test", target_types="category", download=True
    )

    # 只取路径与标签（不触发图片加载），合并后共 7349 条
    all_samples = list(zip(ds_trainval._images, ds_trainval._labels)) + list(
        zip(ds_test._images, ds_test._labels)
    )
    all_labels = [label for _, label in all_samples]

    # 第一次：切出 30% 作为「验证+测试」临时集，剩余 70% 为训练集
    train_samples, temp_samples, _, temp_labels = train_test_split(
        all_samples,
        all_labels,
        test_size=0.30,
        stratify=all_labels,   # 分层：保证每个类别都按相同比例划分
        random_state=seed,
    )

    # 第二次：把 30% 对半切成 15% / 15%
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.50,
        stratify=temp_labels,
        random_state=seed,
    )

    return train_samples, val_samples, test_samples


def get_class_names(root: str = "./data") -> List[str]:
    """返回 37 个品种名（按标签索引排序），供混淆矩阵绘图使用。"""
    ds = OxfordIIITPet(root=root, split="trainval", target_types="category")
    return ds.classes


# ---------- 4. 构建 DataLoader ----------
def build_dataloaders(
    root: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    rand_augment: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    torch.manual_seed(seed)

    train_samples, val_samples, test_samples = stratified_split(root, seed)
    train_tf, eval_tf = build_transforms(rand_augment=rand_augment)

    train_set = PetDataset(train_samples, transform=train_tf)
    val_set = PetDataset(val_samples, transform=eval_tf)
    test_set = PetDataset(test_samples, transform=eval_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,           # 仅训练集打乱
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,          # 评估需可复现
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders()

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"train: {n_train}")
    print(f"val:   {n_val}")
    print(f"test:  {n_test}")
    print(f"total: {n_train + n_val + n_test}  (应为 7349)")
    print(f"classes: {len(get_class_names())}")

    x, y = next(iter(train_loader))
    print("batch:", x.shape, y.shape, x.dtype)
