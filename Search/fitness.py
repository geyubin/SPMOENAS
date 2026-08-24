# encoding: utf-8
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from SearchSpace import *
from dataPrepare import build_train_Optimizer_Loss


def indicator(x_list, y_list):
    # 将输入列表转换为 NumPy 数组
    x_array = np.array(x_list)
    y_array = np.array(y_list)

    # 计算差值
    diff = y_array - x_array

    # 检查每行是否所有元素都满足 diff >= 0
    # result = np.all(diff >= 0, axis=0).astype(int)  # 转换为 1 或 0
    result = [1 if x >= 0 else 0 for x in diff]
    return result


def fitness_(in_, input_train, input_valid, device):
    cur_device = device
    rounded_in = [round(x) for x in in_]
    model = NetworkCifar(rounded_in)
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
    fit_1 = total_params / 1e6

    fit_2 = 100

    return [fit_1, -fit_2]


def evaluate_architecture_without_teachers(in_, input_train, input_valid, min_vals, max_vals, device=None,
                                           num_classes=10, model_channels=64,
                                           model_stem_type="cifar",
                                           search_use_aux=False,
                                           search_aux_weight=0.4,
                                           search_aux_head_type="cifar"):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    arch_array = np.asarray(in_, dtype=float)
    min_vals = np.asarray(min_vals, dtype=float)
    max_vals = np.asarray(max_vals, dtype=float)

    n_classes = np.round(max_vals - min_vals + 1).astype(int)
    intervals = (max_vals - min_vals) / n_classes
    idx = ((arch_array - min_vals) / intervals).astype(int)
    idx = np.clip(idx, 0, n_classes - 1)
    discrete_arch = (min_vals + idx).astype(int)

    num_class = num_classes
    model = NetworkCifar(
        discrete_arch.tolist(),
        num_classes=num_class,
        C=model_channels,
        stem_type=model_stem_type,
        use_aux=bool(search_use_aux),
        aux_head_type=search_aux_head_type,
    )
    model.to(device)

    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
    fit_1 = total_params / 1e6

    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.025,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for inputs, targets in input_train:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            logits, aux_logits = outputs
            loss = criterion(logits, targets) + float(search_aux_weight) * criterion(aux_logits, targets)
        else:
            logits = outputs
            loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in input_valid:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_acc = correct / max(1, total)
    fit_2 = -val_acc
    print(f'candidate objectives: f1(params_M)={fit_1:.6f}, f2(-valid_acc)={fit_2:.6f}')

    return [fit_1, fit_2]
