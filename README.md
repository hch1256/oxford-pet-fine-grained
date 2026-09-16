# 基于深度学习的牛津宠物细粒度图像分类

> 姓名：韩承翰 ｜ 学号：W124301159 ｜ 数据集：Oxford-IIIT Pet（37 类细粒度分类，7,349 张）
> 骨干网络：ResNet-18（ImageNet 预训练）｜ 硬件：NVIDIA RTX 4060 Laptop (8GB)
> 代码仓库：[hch1256/oxford-pet-fine-grained](https://github.com/hch1256/oxford-pet-fine-grained)

基于 ResNet-18 迁移学习实现猫狗品种细粒度分类，并系统对比了 **RandAugment** 与
**Label Smoothing** 两种正则化策略。完整实验在单张消费级 GPU 上约 15 分钟内可复现。

---

## 一、实验结果

所有指标均在**测试集（1,103 张，全程未参与任何训练与模型选择决策）**上测得：

| 实验配置 | 验证集 Top-1 | 测试集 Top-1 | Top-5 Acc | Macro-F1 | 误判数 | 最佳轮次 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1. Baseline（基础增强） | 92.56% | 90.84% | 99.27% | 0.9080 | 101 | ep6 |
| 2. + RandAugment | 92.74% | 90.93% | 99.18% | 0.9084 | 100 | ep7 |
| 3. **+ Label Smoothing (0.1)** | **93.56%** | **92.38%** | 98.91% | **0.9230** | **84** | ep10 |

### 核心结论

1. **RandAugment 效果有限**：测试集 Top-1 仅提升 0.09pp（约 1 个样本），不具统计显著性。
   但其正则化作用明确——训练集 loss 终值由 0.038 升至 0.097（2.5 倍），
   train/val 泛化差距由 0.281 缩小至 0.221（−21.5%），说明模型记忆训练集的能力被有效抑制。

2. **Label Smoothing 效果显著**：测试集 Top-1 提升 **1.54pp**，Macro-F1 提升 0.015，
   误判总数下降 **17%**（101 → 84）。最关键的是 **val_loss 反弹幅度由 +0.058 降至 +0.016**，
   过拟合被大幅抑制。

3. **Label Smoothing 的收益具有选择性**：对「比特犬 ↔ 斯塔福郡斗牛㹴」改善明显（9 → 3 次误判），
   但对「吉娃娃 → 迷你杜宾」完全无效（7 → 7），甚至使「孟加拉猫 ↔ 埃及猫」恶化（8 → 10）。
   说明该方法擅长优化**决策边界模糊**的类别，对**视觉特征本身高度重合**的类别作用有限。

4. **Top-5 与 Top-1 的权衡**：Label Smoothing 的 Top-5 反而略降（99.27% → 98.91%）。
   这是因为软化标签压制了对正确答案的极端自信，部分本可排入前 5 的正确答案被"平均化"挤出——
   属于方法的固有代价，而非缺陷。

> ⚠️ **注意**：三组实验的 loss 绝对值**不可横向比较**。Label Smoothing 改变了损失函数定义
> （目标分布含 0.1 均匀项，理论下界约 `0.1 × ln(37) ≈ 0.36`），量纲与其他两组不同。
> 只能比较准确率、F1 与曲线形状趋势。

---

## 二、目录结构

```text
pet_project/
├── data/
│   ├── dataset.py            # 数据加载、70:15:15 分层划分、Transform
│   └── oxford-iiit-pet/      # 数据集（需自行下载，见第三节）
├── models/
│   └── model.py              # ResNet-18 构建、分类头替换、微调策略
├── utils/
│   ├── metrics.py            # Top-k 准确率、Macro-F1、混淆矩阵绘制
│   └── gradcam.py            # Grad-CAM 热力图生成
├── train.py                  # 训练主入口（argparse 支持全部消融开关）
├── evaluate.py               # 评估与可视化独立脚本
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 三、环境准备

### 1. 创建环境

```bash
conda create -n pet python=3.11 -y
conda activate pet
```

### 2. 安装 PyTorch（CUDA 版）

**必须使用 CUDA 专用源**，否则会装成 CPU 版导致 `torch.cuda.is_available()` 返回 `False`：

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
```

> CUDA 版本按自己的驱动选择（`cu126` / `cu124` / `cu118`），用 `nvidia-smi` 查看驱动支持的最高版本。
> 国内网络可用镜像加速：`--index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126/`

### 3. 安装其余依赖

```bash
pip install -r requirements.txt
```

### 4. 验证 GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

应输出 `True` 和显卡型号。

### 5. 下载数据集

`torchvision` 会在首次运行 `train.py` 时自动下载（约 800MB）。也可手动下载后
解压到 `data/oxford-iiit-pet/`，最终目录结构须为：

```text
data/oxford-iiit-pet/
├── images/          # 7,390 张 .jpg
└── annotations/     # list.txt / trainval.txt / test.txt / trimaps/ / xmls/
```

验证数据就位：

```bash
python -c "from torchvision.datasets import OxfordIIITPet; print(len(OxfordIIITPet(root='./data', download=True, split='trainval')))"
# 应输出 3680，且不出现下载进度条
```

---

## 四、一键复现

```bash
# 1. 训练基线（约 15 分钟）
python train.py --amp --out_dir ./runs/baseline

# 2. 评估并生成全部可视化结果
python evaluate.py --ckpt ./runs/baseline/best_model.pth

# 3. 查看训练曲线
python -m tensorboard.main --logdir ./runs --port 6006
# 浏览器打开 http://localhost:6006
```

---

## 五、消融实验

**单变量控制原则：每组实验仅修改一个参数，其余保持默认。**

```bash
# 实验 2：进阶数据增强
python train.py --amp --randaugment --out_dir ./runs/randaug

# 实验 3：Label Smoothing
python train.py --amp --label_smoothing 0.1 --out_dir ./runs/ls01

# 可选实验 4：仅微调分类头（冻结骨干）
python train.py --amp --freeze_backbone --out_dir ./runs/frozen

# 可选实验 5：MixUp
python train.py --amp --mixup 0.2 --out_dir ./runs/mixup
```

常用参数：

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--epochs` | 15 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--optimizer` | adamw | `adamw` 或 `sgd` |
| `--amp` | 关闭 | 混合精度训练，约提速 20~40% |
| `--seed` | 42 | 随机种子 |
| `--num_workers` | 4 | DataLoader 子进程数 |

---

## 六、输出文件说明

每个实验的输出目录（如 `runs/baseline/`）包含：

| 文件 | 内容 |
| :--- | :--- |
| `best_model.pth` | 验证集最优权重（含训练配置，供 `evaluate.py` 自动读取） |
| `history.json` | 每轮的 train_loss / val_loss / val_acc / macro_f1 / 耗时 |
| `events.out.tfevents.*` | TensorBoard 日志 |
| `metrics.json` | 测试集 Top-1 / Top-5 / Macro-F1 |
| `confusion_matrix.png` | 37 类混淆矩阵（计数） |
| `confusion_matrix_norm.png` | 37 类混淆矩阵（行归一化） |
| `gradcam_correct.png` | 预测正确样本的 Grad-CAM |
| `gradcam_error.png` | 预测错误样本的 Grad-CAM |
| `gradcam_compare.png` | 两者并排对比图 |
| `misclassified.txt` | 全部误判样本清单（真实标签 / 预测标签 / 置信度 / 文件名） |

---

## 七、关键实现说明

**数据处理**：合并 torchvision 自带的 `trainval`(3680) 与 `test`(3669) 两个 split 成全量 7,349 张，
再用 `sklearn.model_selection.train_test_split` 做两步分层抽样得到 70:15:15
（5144 / 1102 / 1103），`random_state=42` 保证可复现。

**延迟加载**：`PetDataset` 只保存 `(图片路径, 标签)`，在 `__getitem__` 中才打开图片。
这对 `num_workers > 0` 至关重要——DataLoader 会把数据集对象 pickle 复制给每个子进程，
若预先加载全部图片（约 4~5GB），多进程会直接耗尽内存。

**Classifier Head 替换**：`model.fc = nn.Linear(512, 37)`，仅替换最后一层，
保留全部预训练卷积权重。可训练参数量由 11,195,493 降至 18,981（`--freeze_backbone` 时）。

**输出层无 Softmax**：`nn.CrossEntropyLoss` 内部已包含 `log_softmax`，
模型输出裸 logits，额外加 `nn.Softmax()` 会导致梯度消失、训练停滞。

**验证集确定性**：验证/测试集 Transform 仅含 `Resize + CenterCrop + Normalize`，
**不含** `RandomCrop` / `RandomHorizontalFlip`，避免评估时的随机干扰与数据泄漏。

**Windows 兼容**：`train.py` 与 `evaluate.py` 的主逻辑均包裹在 `if __name__ == "__main__":` 中
——Windows 采用 spawn 方式创建子进程，缺少该保护会导致 `num_workers > 0` 时无限递归。

---

## 八、AI 辅助编程声明

本项目在开发过程中使用了 AI 编程工具辅助，主要用途：
生成 Grad-CAM 可视化代码、编写 TensorBoard 记录逻辑、以及代码结构建议。

**AI 生成代码中发现并修复的 Bug**（真实记录）：

1. **数据加载内存爆炸**：AI 最初生成的代码用 `list(dataset)` 合并两个 split，
   会把全部 7,349 张图片解码进内存（约 4~5GB），配合 `num_workers=4` 时
   DataLoader 会为每个子进程复制一份，总内存需求超过 20GB 导致进程崩溃。
   **修复**：改为只保存 `(路径, 标签)` 元组，在 `__getitem__` 中延迟打开图片。

2. **混淆矩阵格式化崩溃**：绘制混淆矩阵时先执行了 `cm.astype(np.float64)`，
   随后 `seaborn.heatmap` 用 `fmt="d"`（整数格式）渲染浮点数组，
   抛出 `ValueError: Unknown format code 'd' for object of type 'float'`。
   **修复**：仅在归一化分支转浮点，计数分支保持整数类型。

3. **中文标题渲染为方框**：matplotlib 默认字体 DejaVu Sans 不含中文字形，
   图中所有中文标题显示为豆腐块。
   **修复**：设置 `plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]`。
