import argparse
import logging
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from SearchSpace import NetworkCifar
from dataPrepare import build_cinic10_train_valid_test_loader


def count_parameters_in_MB(model):
    return sum(v.numel() for v in model.parameters()) / 1e6


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(model_name, epochs, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f'{model_name}_{epochs}.log')

    logger = logging.getLogger('cinic10_train')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger, log_filename


def build_loss_optimizer_scheduler(model, args, device):
    try:
        train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
        eval_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    except TypeError:
        train_criterion = nn.CrossEntropyLoss().to(device)
        eval_criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr_max,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    warmup_epochs = min(max(0, int(args.warmup_epochs)), max(0, int(args.epochs) - 1))
    warmup_start_factor = min(1.0, max(1e-8, float(args.warmup_start_factor)))
    eta_min_factor = float(args.lr_min) / float(args.lr_max) if args.lr_max > 0 else 0.0

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return warmup_start_factor + (1.0 - warmup_start_factor) * float(epoch) / float(warmup_epochs)
        cosine_epochs = max(1, int(args.epochs) - warmup_epochs)
        progress = min(1.0, max(0.0, float(epoch - warmup_epochs) / float(cosine_epochs)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_factor + (1.0 - eta_min_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return train_criterion, eval_criterion, optimizer, scheduler


def smooth_one_hot(labels, num_classes, smoothing):
    off_value = float(smoothing) / float(num_classes)
    on_value = 1.0 - float(smoothing) + off_value
    targets = torch.full(
        (labels.size(0), num_classes),
        off_value,
        device=labels.device,
        dtype=torch.float32,
    )
    targets.scatter_(1, labels.view(-1, 1), on_value)
    return targets


def soft_target_cross_entropy(logits, targets):
    return torch.sum(-targets * F.log_softmax(logits, dim=1), dim=1).mean()


def rand_bbox(height, width, lam):
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)
    cy = np.random.randint(height)
    cx = np.random.randint(width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    x1 = np.clip(cx - cut_w // 2, 0, width)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    return y1, y2, x1, x2


def apply_mixup_cutmix(inputs, labels, args, num_classes):
    if args.mix_prob <= 0.0 or random.random() > args.mix_prob:
        return inputs, labels, False

    use_mixup = args.mixup_alpha > 0.0
    use_cutmix = args.cutmix_alpha > 0.0
    if not use_mixup and not use_cutmix:
        return inputs, labels, False

    if use_mixup and use_cutmix:
        do_cutmix = random.random() < args.cutmix_switch_prob
    else:
        do_cutmix = use_cutmix

    batch_size = inputs.size(0)
    perm = torch.randperm(batch_size, device=inputs.device)
    labels_a = smooth_one_hot(labels, num_classes, args.label_smoothing)
    labels_b = smooth_one_hot(labels[perm], num_classes, args.label_smoothing)

    if do_cutmix:
        lam = np.random.beta(args.cutmix_alpha, args.cutmix_alpha)
        y1, y2, x1, x2 = rand_bbox(inputs.size(2), inputs.size(3), lam)
        mixed_inputs = inputs.clone()
        mixed_inputs[:, :, y1:y2, x1:x2] = inputs[perm, :, y1:y2, x1:x2]
        box_area = float((y2 - y1) * (x2 - x1))
        lam = 1.0 - box_area / float(inputs.size(2) * inputs.size(3))
    else:
        lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
        mixed_inputs = inputs * lam + inputs[perm] * (1.0 - lam)

    mixed_targets = labels_a * lam + labels_b * (1.0 - lam)
    return mixed_inputs, mixed_targets, True


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            predicted = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / max(1, total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CINIC-10 Architecture')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device ID (-1 for CPU)')
    parser.add_argument('--aux_weight', type=float, default=0.4, help='Auxiliary head loss weight')
    parser.add_argument('--epochs', type=int, default=600, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--use_aux', type=int, default=1, choices=[0, 1], help='Use auxiliary head (1) or not (0)')
    parser.add_argument('--lr_max', type=float, default=0.025, help='Maximum learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing factor')
    parser.add_argument('--model_name', type=str, default='genotype', help='Model architecture name')
    parser.add_argument('--use_cutout', type=int, default=1, choices=[0, 1], help='Use Cutout augmentation')
    parser.add_argument('--train_with_valid', type=int, default=0, choices=[0, 1],
                        help='Add CINIC-10 valid split to training data for final training')
    parser.add_argument('--model_channels', type=int, default=64, help='Initial channel count of NetworkCifar')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Classifier dropout rate')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Linear warmup epochs')
    parser.add_argument('--warmup_start_factor', type=float, default=0.1, help='Warmup start LR factor')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='Max gradient norm, <=0 disables clipping')
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1], help='Use automatic mixed precision')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup beta distribution alpha')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='CutMix beta distribution alpha')
    parser.add_argument('--mix_prob', type=float, default=0.5, help='Probability of applying mixup or CutMix')
    parser.add_argument('--cutmix_switch_prob', type=float, default=0.5, help='CutMix choice probability when both are enabled')
    parser.add_argument('--drop_path_prob', type=float, default=0.1, help='Final drop path probability')
    parser.add_argument('--eval_train', type=int, default=1, choices=[0, 1], help='Evaluate train accuracy each epoch')
    parser.add_argument('--print_freq', type=int, default=1, help='Log every N epochs')
    parser.add_argument('--result_dir', type=str, default='TrainResults_CINIC10', help='Result directory')
    parser.add_argument('--log_dir', type=str, default='TrainResults_CINIC10/logs', help='Log directory')
    args = parser.parse_args()

    os.environ['USE_CUTOUT'] = '1' if bool(args.use_cutout) else '0'

    set_seed(seed=args.seed)

    # 模型名称
    model_name = args.model_name

    logger, log_filename = setup_logger(model_name, args.epochs, args.log_dir)

    # 记录开始时间
    start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"Training Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # 打印所有参数
    logger.info("Training Configuration:")
    logger.info(f"  Model Name: {model_name}")
    logger.info(f"  CUDA Device: {args.cuda}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Use Auxiliary Head: {bool(args.use_aux)}")
    logger.info(f"  Auxiliary Weight: {args.aux_weight}")
    logger.info(f"  Max Learning Rate: {args.lr_max}")
    logger.info(f"  Momentum: {args.momentum}")
    logger.info(f"  Weight Decay: {args.weight_decay}")
    logger.info(f"  Label Smoothing: {args.label_smoothing}")
    logger.info(f"  Use Cutout: {bool(args.use_cutout)}")
    logger.info(f"  Train With Valid: {bool(args.train_with_valid)}")
    logger.info(f"  Model Channels: {args.model_channels}")
    logger.info(f"  Dropout Rate: {args.dropout_rate}")
    logger.info(f"  LR Min: {args.lr_min}")
    logger.info(f"  Warmup Epochs: {args.warmup_epochs}")
    logger.info(f"  Grad Clip: {args.grad_clip}")
    logger.info(f"  AMP: {bool(args.amp)}")
    logger.info(f"  Mixup Alpha: {args.mixup_alpha}")
    logger.info(f"  CutMix Alpha: {args.cutmix_alpha}")
    logger.info(f"  Mix Prob: {args.mix_prob}")
    logger.info(f"  Drop Path Prob: {args.drop_path_prob}")
    logger.info("=" * 80)

    train_accuracy = []
    valid_accuracy = []
    test_accuracy = []

    # 定义可用的模型架构
    model_architectures = {   
      'genotype':  [1, 1, 0, 0, 0, 6, 6, 0, 1, 0, 4, 4, 0, 2, 1, 4, 2, 0, 2, 1, 6, 5, 0, 2, 2, 3, 5, 1, 1, 1, 6, 6, 1, 2, 1, 4, 2, 0, 2, 2, 6, 4, 0, 0, 2, 2, 4, 0, 2, 1, 0, 4, 0, 1, 0, 3, 4, 0, 1, 0, 4, 5, 0, 2, 1, 6, 5, 1, 3, 1, 2, 2, 0, 2, 1, 5, 3, 0, 0, 2, 0, 1, 1, 2, 0, 3, 1, 2, 1, 1, 1, 1, 0, 2, 0, 3, 1, 2],
    }

    # 根据model_name获取对应的架构，如果不存在则使用默认
    if model_name in model_architectures:
        genotype = model_architectures[model_name]
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Available architectures: {list(model_architectures.keys())}")

    aux_weight = args.aux_weight
    use_aux = bool(args.use_aux)

    if args.cuda < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    train_loader, valid_loader, test_loader = build_cinic10_train_valid_test_loader(
        args.batch_size,
        include_valid_in_train=bool(args.train_with_valid)
    )
    logger.info(f"Train Samples: {len(train_loader.dataset)}")
    logger.info(f"Valid Samples: {len(valid_loader.dataset)}")
    logger.info(f"Test Samples: {len(test_loader.dataset)}")

    model = NetworkCifar(
        genotype,
        num_classes=10,
        C=args.model_channels,
        use_aux=use_aux,
        dropout_rate=args.dropout_rate,
        drop_path_prob=0.0
    ).to(device)

    model_params = count_parameters_in_MB(model)
    logger.info(f'Model Parameters: {model_params} MB')

    epochs = args.epochs

    train_criterion, eval_criterion, optimizer, scheduler = build_loss_optimizer_scheduler(model, args, device)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == 'cuda')

    logger.info("Starting training loop...")
    logger.info("-" * 80)

    best_test_acc = -1.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()

        model.train()
        model.drop_path_prob = args.drop_path_prob * epoch / max(1, epochs - 1)
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            inputs, targets, soft_targets = apply_mixup_cutmix(inputs, labels, args, num_classes=10)
            optimizer.zero_grad(set_to_none=True)

            use_amp = bool(args.amp) and device.type == 'cuda'
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    logits, aux_logits = outputs
                    if soft_targets:
                        loss = soft_target_cross_entropy(logits, targets)
                        loss = loss + aux_weight * soft_target_cross_entropy(aux_logits, targets)
                    else:
                        loss = train_criterion(logits, targets) + aux_weight * train_criterion(aux_logits, targets)
                else:
                    if soft_targets:
                        loss = soft_target_cross_entropy(outputs, targets)
                    else:
                        loss = train_criterion(outputs, targets)

            if use_amp:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        train_acc = evaluate(model, train_loader, device) if bool(args.eval_train) else float('nan')
        valid_acc = evaluate(model, valid_loader, device)
        test_acc = evaluate(model, test_loader, device)
        train_accuracy.append(train_acc)
        valid_accuracy.append(valid_acc)
        test_accuracy.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if args.print_freq > 0 and (epoch + 1) % args.print_freq == 0:
            logger.info(f'Epoch [{epoch+1}/{epochs}] - '
                       f'Loss: {avg_train_loss:.4f}, '
                       f'LR: {current_lr:.6f}, '
                       f'DropPath: {model.drop_path_prob:.4f}, '
                       f'Train Acc: {train_acc:.2f}%, '
                       f'Valid Acc: {valid_acc:.2f}%, '
                       f'Test Acc: {test_acc:.2f}%, '
                       f'Best Test: {best_test_acc:.2f}% (Epoch {best_epoch}), '
                       f'Time: {epoch_duration:.2f}s')

    # 训练结束
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    logger.info("-" * 80)
    logger.info("Training Completed!")
    logger.info(f"Total Training Time: {hours}h {minutes}m {seconds}s")
    logger.info(f"Final Test Accuracy: {test_accuracy[-1]:.2f}%")
    logger.info(f"Best Test Accuracy: {best_test_acc:.2f}% at epoch {best_epoch}")
    logger.info("=" * 80)

    # 保存结果文件
    file_name = os.path.join(args.result_dir, f'train_cinic10_accuracy_{args.model_name}_{args.drop_path_prob}.txt')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, train_accuracy)))

    file_name = os.path.join(args.result_dir, f'valid_cinic10_accuracy_{args.model_name}_{args.drop_path_prob}.txt')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, valid_accuracy)))

    file_name = os.path.join(args.result_dir, f'test_cinic10_accuracy_{args.model_name}_{args.drop_path_prob}.txt')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, test_accuracy)))

    logger.info(f"Results saved to {args.result_dir}/")
