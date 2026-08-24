# MOPSO-NAS

基于多目标粒子群优化（MOPSO）的神经网络架构搜索与图像分类训练项目，支持 CIFAR-10、CIFAR-100、CINIC-10、Tiny-ImageNet 和 ImageNet。

## 环境安装

建议使用 Conda 创建独立环境，然后安装依赖：

```bash
pip install numpy torch torchvision pillow matplotlib pygraphviz
```

## 数据集

- CIFAR-10/CIFAR-100：程序可通过 `torchvision` 自动下载。
- CINIC-10：默认放在 `./data/cinic10/`。
- Tiny-ImageNet：默认放在 `./data/tiny-imagenet-200/`。
- ImageNet：通过 `--imagenet_root` 指定数据集目录。

## 架构搜索

```bash
python NewMOPSO/main.py --dataset cifar10 --cuda 0
```

可将 `--dataset` 设置为 `cifar10`、`cifar100`、`cinic10`、`tiny` 或 `imagenet`。

## 模型训练

```bash
python trainforcifar10.py --cuda 0
python trainforcifar100.py --cuda 0
python trainforCINIC10.py --cuda 0
python trainforTinyImagenet.py --cuda 0
```

使用 `--help` 查看各脚本的完整参数：

```bash
python NewMOPSO/main.py --help
```

## 项目结构

```text
NewMOPSO/               MOPSO 搜索算法
SearchSpace.py          神经网络搜索空间
Operation.py            网络基本操作
dataPrepare.py          数据集与优化器配置
trainfor*.py            各数据集训练入口
```

数据集、训练日志、模型权重及缓存文件不应提交到 GitHub。
