# models/model.py
"""ResNet-18 骨干网络的构建、分类头替换与微调策略控制。

设计要点：
1. 用 ImageNet 预训练权重做迁移学习——37 类细粒度分类只有 5k 张训练图，
   从零训练会严重过拟合。
2. 替换最后的全连接层，把 ImageNet 的 1000 类输出改成 37 类。
   注意：只换最后一层，前面的卷积层保留预训练权重。
3. **输出层不加 nn.Softmax()**：nn.CrossEntropyLoss 内部已包含 log_softmax，
   再加一层会导致数值不稳定且梯度错误。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 37


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """构建用于细粒度分类的 ResNet-18。

    Args:
        num_classes: 输出类别数，Oxford-IIIT Pet 为 37。
        pretrained: 是否加载 ImageNet 预训练权重。
        freeze_backbone: 为 True 时冻结除 fc 外的全部参数，仅训练分类头。

    Returns:
        改造好的 nn.Module。
    """
    # 新版 torchvision 推荐用 Weights 枚举而非弃用的 pretrained=True
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # 替换分类头：ResNet-18 的 fc 输入维度是 512
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 微调策略：冻结骨干，只训练新换上的 fc 层
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """返回 Grad-CAM 要挂载的目标层：最后一个残差块。

    选它是因为该层输出的特征图仍有 7x7 的空间分辨率，
    既有高层语义信息，又能定位到图像的具体区域。
    """
    return model.layer4[-1]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (可训练参数量, 总参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


if __name__ == "__main__":
    # 1. 检查分类头维度
    model = build_model()
    print(f"fc.in_features = {model.fc.in_features}  (应为 512)")
    print(f"fc.out_features = {model.fc.out_features}  (应为 37)")

    # 2. 前向传播，检查输入输出形状
    dummy = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"input:  {tuple(dummy.shape)}")
    print(f"output: {tuple(out.shape)}  (应为 [2, 37])")

    # 3. 检查 Grad-CAM 目标层的特征图形状（答辩考点）
    # 用 forward hook 抓取中间层输出——Grad-CAM 就是靠这个拿到特征图的
    target_layer = get_gradcam_target_layer(model)
    captured: dict[str, torch.Tensor] = {}
    handle = target_layer.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("feat", output)
    )
    with torch.no_grad():
        model(dummy)
    handle.remove()
    print(f"layer4[-1] 输出: {tuple(captured['feat'].shape)}  (应为 [2, 512, 7, 7])")

    # 4. 检查冻结策略
    full = build_model(freeze_backbone=False)
    frozen = build_model(freeze_backbone=True)
    print(f"全参数微调  可训练参数: {count_parameters(full)[0]:,}")
    print(f"仅微调 fc   可训练参数: {count_parameters(frozen)[0]:,}")
