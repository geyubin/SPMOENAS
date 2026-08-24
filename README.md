# SPMOENAS

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


## 项目结构

```text
Search/                 搜索策略
SearchSpace.py          神经网络搜索空间
Operation.py            网络基本操作
dataPrepare.py          数据集与优化器配置
trainfor*.py            各数据集训练入口
```

数据集、训练日志、模型权重及缓存文件不应提交到 GitHub。
