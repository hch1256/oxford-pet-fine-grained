# utils/gradcam.py
"""Grad-CAM 热力图可视化。

原理（答辩会问）：
1. 取目标层（通常是最后一个残差块 layer4[-1]）的**前向激活图** A，形状 [C, H, W]；
2. 对同一层求**反向梯度** dY/dA；
3. 把梯度在每个通道上做全局平均，得到 C 个权重 alpha_c；
   它衡量「第 c 个特征图对最终分类结果有多重要」；
4. 加权求和 + ReLU：L = ReLU(sum_c alpha_c * A_c)，得到单通道热力图；
5. 热力图上采样回原图尺寸，叠加显示。

为什么选 layer4[-1]：它同时具备**高层语义**（认得整只猫/狗）和
**空间分辨率**（7x7，能定位到图像的具体区域）。如果再往后到 avgpool，
空间维度就没了，无法定位。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 grad-cam 库，请先运行：pip install grad-cam"
    ) from exc

# 与 data/dataset.py 中保持一致，用于把归一化张量还原成可显示图像
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """把归一化后的 [3, H, W] Tensor 还原成 float32 [0,1] 的 HWC 图像。

    Normalize 做的是 (x - mean) / std，所以逆运算就是 x * std + mean。
    """
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def generate_gradcam(
    model: torch.nn.Module,
    image_path: Path | str,
    transform: Callable,
    target_layer: torch.nn.Module,
    device: torch.device,
    target_class: Optional[int] = None,
) -> Dict[str, object]:
    """对单张图片生成 Grad-CAM 热力图。

    Args:
        target_class: 要解释哪一个类别的判断。默认为模型预测的那个类；
                      分析误判样本时可传**真实类别**，看模型为什么没注意到关键区域。

    Returns:
        dict，含 overlay（叠加热力图，HWC uint8）、rgb（原图 HWC uint8）、
        pred、conf、true_prob 等字段。
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # 先拿到预测结果与置信度
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    pred = int(logits.argmax(dim=1).item())
    conf = float(probs[pred].item())

    if target_class is None:
        target_class = pred

    # Grad-CAM 内部需要前向 + 反向，这里不能包在 no_grad 里
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class)],
    )[0]

    rgb_float = denormalize(input_tensor[0])           # 模型实际「看到」的画面
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)  # HWC uint8

    return {
        "overlay": overlay,
        "rgb": (rgb_float * 255).astype(np.uint8),
        "pred": pred,
        "conf": conf,
        "target_class": target_class,
        "true_prob": float(probs[target_class].item()),
    }
