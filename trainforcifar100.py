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
from dataPrepare import build_cifar100_train_valid_test_loader


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

    logger = logging.getLogger('cifar100_train')
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


def get_architectures():
    genotype =  [1, 0, 0, 0, 1, 6, 5, 0, 1, 0, 2, 5, 0, 1, 3, 5, 0, 1, 3, 0, 5, 4, 0, 1, 4, 4, 1, 0, 1, 0, 5, 6, 0, 2, 1, 4, 5, 0, 2, 1, 4, 1, 0, 2, 1, 1, 1, 0, 2, 2, 4, 2, 0, 1, 1, 6, 1, 1, 0, 1, 6, 4, 1, 2, 2, 4, 3, 0, 0, 1, 3, 6, 1, 2, 3, 4, 5, 0, 0, 0, 0, 1, 1, 4, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 0, 2, 1, 2]

    return {
        'genotype': genotype,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR-100 Architecture')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device ID (-1 for CPU)')
    parser.add_argument('--model_name', type=str, default='genotype', help='Architecture name')
    parser.add_argument('--indice', type=int, default=None, help='Deprecated architecture index, kept for old runs')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--use_aux', type=int, default=1, choices=[0, 1])
    parser.add_argument('--aux_weight', type=float, default=0.4)
    parser.add_argument('--lr_max', type=float, default=0.025)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_start_factor', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--grad_clip', type=float, default=2.0)
    parser.add_argument('--amp', type=int, default=1, choices=[0, 1])
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--mix_prob', type=float, default=0.5)
    parser.add_argument('--cutmix_switch_prob', type=float, default=0.5)
    parser.add_argument('--model_channels', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--drop_path_prob', type=float, default=0.1)
    parser.add_argument('--eval_train', type=int, default=1, choices=[0, 1])
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='TrainResults_cifar100')
    parser.add_argument('--log_dir', type=str, default='TrainResults_cifar100/logs')
    return parser.parse_args()


def resolve_model_name(args):
    indice_map = {
        1: 'ablation_A_cifar100',
        2: 'ablation_B_cifar100',
        3: 'ablation_C_cifar100',
        4: 'ratio_25_cifar100',
        5: 'ratio_50_cifar100',
        6: 'ratio_75_cifar100',
        7: 'ratio_100_cifar100',
        8: 'mopso_cifar100',
        9: 'H',
    }
    if args.indice is not None:
        if args.indice not in indice_map:
            raise ValueError(f"indice must be one of {sorted(indice_map)}, got {args.indice}")
        return indice_map[args.indice]
    return args.model_name


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            predicted = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / max(1, total)


def main():
    args = parse_args()
    model_name = resolve_model_name(args)
    architectures = get_architectures()
    if model_name not in architectures:
        raise ValueError(f"Unknown model_name '{model_name}'. Available architectures: {list(architectures.keys())}")

    set_seed(args.seed)
    genotype = architectures[model_name]
    logger, log_filename = setup_logger(model_name, args.epochs, args.log_dir)
    start_time = time.time()

    if args.cuda < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_cifar100_train_valid_test_loader(args.batch_size)
    model = NetworkCifar(
        genotype,
        num_classes=100,
        C=args.model_channels,
        use_aux=bool(args.use_aux),
        dropout_rate=args.dropout_rate,
        drop_path_prob=0.0,
    ).to(device)

    train_criterion, eval_criterion, optimizer, scheduler = build_loss_optimizer_scheduler(model, args, device)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == 'cuda')

    logger.info("=" * 80)
    logger.info(f"CIFAR-100 Training Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_filename}")
    for key, value in sorted(vars(args).items()):
        logger.info(f"{key}: {value}")
    logger.info(f"resolved_model_name: {model_name}")
    logger.info(f"device: {device}")
    logger.info(f"genotype_length: {len(genotype)}")
    logger.info(f"genotype: {genotype}")
    logger.info(f"train_samples: {len(train_loader.dataset)}")
    logger.info(f"test_samples: {len(test_loader.dataset)}")
    logger.info(f"model_parameters_mb: {count_parameters_in_MB(model):.6f}")
    logger.info(f"optimizer: {optimizer.__class__.__name__}")
    logger.info(f"scheduler: {scheduler.__class__.__name__}")
    logger.info("=" * 80)

    train_accuracy = []
    test_accuracy = []
    best_test_acc = -1.0
    best_epoch = 0

    os.makedirs(args.result_dir, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        model.drop_path_prob = args.drop_path_prob * epoch / max(1, args.epochs - 1)
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            inputs, targets, soft_targets = apply_mixup_cutmix(inputs, labels, args, num_classes=100)
            optimizer.zero_grad(set_to_none=True)

            use_amp = bool(args.amp) and device.type == 'cuda'
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    logits, aux_logits = outputs
                    if soft_targets:
                        loss = soft_target_cross_entropy(logits, targets)
                        loss = loss + args.aux_weight * soft_target_cross_entropy(aux_logits, targets)
                    else:
                        loss = train_criterion(logits, targets) + args.aux_weight * train_criterion(aux_logits, targets)
                else:
                    if soft_targets:
                        loss = soft_target_cross_entropy(outputs, targets)
                    else:
                        loss = train_criterion(outputs, targets)

            if scaler is not None and use_amp:
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

        avg_loss = running_loss / max(1, len(train_loader))
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        train_acc = evaluate(model, train_loader, device) if bool(args.eval_train) else float('nan')
        test_acc = evaluate(model, test_loader, device)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1

        if args.print_freq > 0 and (epoch + 1) % args.print_freq == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"Loss: {avg_loss:.4f}, "
                f"LR: {current_lr:.6f}, "
                f"DropPath: {model.drop_path_prob:.4f}, "
                f"Train Acc: {train_acc:.2f}%, "
                f"Test Acc: {test_acc:.2f}%, "
                f"Best Test: {best_test_acc:.2f}% (Epoch {best_epoch}), "
                f"Time: {time.time() - epoch_start:.2f}s"
            )

    total_time = time.time() - start_time
    logger.info("-" * 80)
    logger.info("Training Completed!")
    logger.info(f"Total Training Time: {total_time / 3600.0:.2f}h")
    logger.info(f"Final Test Accuracy: {test_accuracy[-1]:.2f}%")
    logger.info(f"Best Test Accuracy: {best_test_acc:.2f}% at epoch {best_epoch}")

    with open(os.path.join(args.result_dir, f'train_cifar100_accuracy_{model_name}_{args.drop_path_prob}.txt'), 'w') as f:
        f.write(", ".join(map(str, train_accuracy)))
    with open(os.path.join(args.result_dir, f'test_cifar100_accuracy_{model_name}_{args.drop_path_prob}.txt'), 'w') as f:
        f.write(", ".join(map(str, test_accuracy)))
    logger.info(f"Results saved to {args.result_dir}/")


if __name__ == '__main__':
    main()
