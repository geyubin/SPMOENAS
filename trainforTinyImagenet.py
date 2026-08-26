import argparse
import logging
import os
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps

from SearchSpace import *
from dataPrepare import build_train_Optimizer_Loss, build_train_tiny_imagenet_train_val_loader


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


def setup_logger(arch_name, epochs):
    log_dir = 'TrainResults_TinyImagenet/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f'{arch_name}_{epochs}.log')

    logger = logging.getLogger('tinyimagenet_train')
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


def print_training_config(args, arch_name, genotype, device, model, train_loader, test_loader,
                          optimizer, scheduler, logger):
    logger.info("=" * 80)
    logger.info("Tiny-ImageNet Training Configuration")
    logger.info("=" * 80)
    for key, value in sorted(vars(args).items()):
        logger.info(f"{key}: {value}")
    logger.info(f"device: {device}")
    logger.info(f"arch_name: {arch_name}")
    logger.info(f"genotype_length: {len(genotype)}")
    logger.info(f"genotype: {genotype}")
    logger.info(f"train_samples: {len(train_loader.dataset)}")
    logger.info(f"test_samples: {len(test_loader.dataset)}")
    logger.info(f"train_batches: {len(train_loader)}")
    logger.info(f"test_batches: {len(test_loader)}")
    logger.info(f"model_parameters_mb: {count_parameters_in_MB(model):.6f}")
    logger.info(f"use_aux: {bool(args.use_aux)}")
    if bool(args.use_aux):
        logger.info(f"aux_head_type: {args.aux_head_type}")
        logger.info(f"aux_head_class: {model.aux_head.__class__.__name__}")
    logger.info(f"optimizer: {optimizer.__class__.__name__}")
    for idx, group in enumerate(optimizer.param_groups):
        logger.info(f"optimizer_group_{idx}_lr: {group.get('lr')}")
        logger.info(f"optimizer_group_{idx}_momentum: {group.get('momentum')}")
        logger.info(f"optimizer_group_{idx}_weight_decay: {group.get('weight_decay')}")
        logger.info(f"optimizer_group_{idx}_nesterov: {group.get('nesterov')}")
    logger.info(f"scheduler: {scheduler.__class__.__name__}")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Tiny-ImageNet Architecture')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device ID (-1 for CPU)')
    parser.add_argument('--aux_weight', type=float, default=0.4, help='Auxiliary head loss weight')
    parser.add_argument('--aux_head_type', type=str, default='image', choices=['image', 'cifar'],
                        help='Auxiliary head type')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--use_aux', type=int, default=1, choices=[0, 1], help='Use auxiliary head (1) or not (0)')
    parser.add_argument('--lr_max', type=float, default=0.025, help='Maximum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Linear warmup epochs before cosine decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='Gradient clipping max norm')
    parser.add_argument('--drop_path_prob', type=float, default=0.15, help='Final DropPath probability')
    parser.add_argument('--print_freq', type=int, default=10, help='Epoch interval for printing metrics')
    parser.add_argument('--tiny_root', type=str, default=os.environ.get('TINY_ROOT', './data/tiny-imagenet-200'),
                        help='Tiny-ImageNet dataset root')
    parser.add_argument('--arch', type=str, default="genotype", help='Architecture name')

    args = parser.parse_args()

    logger, log_filename = setup_logger(args.arch, args.epochs)
    start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"Training Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_filename}")
    logger.info("=" * 80)

    set_seed(seed=args.seed)

    train_accuracy = []
    test_accuracy = []

    genotype =  [1, 0, 0, 1, 1, 4, 4, 1, 2, 1, 4, 6, 0, 1, 3, 6, 3, 0, 1, 1, 4, 3, 0, 1, 4, 4, 5, 0, 0, 0, 3, 4, 0, 1, 0, 6, 5, 1, 3, 3, 3, 3, 0, 3, 2, 4, 7, 0, 2, 2, 5, 1, 0, 0, 1, 7, 1, 1, 1, 2, 6, 5, 1, 1, 1, 6, 4, 1, 4, 3, 3, 4, 0, 3, 3, 5, 4, 0, 0, 3, 0, 3, 0, 4, 1, 1, 0, 2, 0, 1, 1, 3, 0, 1, 1, 3, 1, 1]

    architectures = {
        "genotype": genotype,
    }

    if args.arch not in architectures:
        raise ValueError(f"Unknown arch '{args.arch}'. Available architectures: {list(architectures.keys())}")

    genotype = architectures[args.arch]

    aux_weight = args.aux_weight
    use_aux = bool(args.use_aux)

    if args.cuda < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_train_tiny_imagenet_train_val_loader(args.tiny_root, args.batch_size)

    model = NetworkCifar(
        genotype,
        num_classes=200,
        use_aux=use_aux,
        aux_head_type=args.aux_head_type,
        drop_path_prob=0.0,
    ).to(device)
    epochs = args.epochs

    train_criterion, eval_criterion, optimizer, scheduler = build_train_Optimizer_Loss(
        model, args.momentum, args.lr_max, args.weight_decay,
        epochs, device=device, label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs
    )

    print_training_config(
        args, args.arch, genotype, device, model, train_loader, test_loader,
        optimizer, scheduler, logger
    )

    for epoch in range(epochs):
        model.train()
        model.drop_path_prob = args.drop_path_prob * epoch / max(1, epochs - 1)
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits, aux_logits = outputs
                loss_main = train_criterion(logits, labels)
                loss_aux = train_criterion(aux_logits, labels)
                loss = loss_main + aux_weight * loss_aux
                preds = torch.max(logits.data, 1)[1]
            else:
                loss = train_criterion(outputs, labels)
                preds = torch.max(outputs.data, 1)[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            running_loss += loss.item()
        if args.print_freq > 0 and epoch % args.print_freq == 0:
            logger.info(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        total_train = 0
        correct_train = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                _, predicted = torch.max(logits.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if args.print_freq > 0 and epoch % args.print_freq == 0:
            logger.info(f'Accuracy on the Tiny-ImageNet train images: {100 * correct_train / total_train:.2f}%')
            logger.info(f'Accuracy on the Tiny-ImageNet test images: {100 * correct / total:.2f}%')
        train_accuracy.append(100 * correct_train / total_train)
        test_accuracy.append(100 * correct / total)

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    logger.info("-" * 80)
    logger.info("Training Completed!")
    logger.info(f"Total Training Time: {hours}h {minutes}m {seconds}s")
    logger.info(f"Final Test Accuracy: {test_accuracy[-1]:.2f}%")
    logger.info(f"Best Test Accuracy: {max(test_accuracy):.2f}%")
    logger.info("=" * 80)

    file_name = f'TrainResults_TinyImagenet/train_tinyimagenet_accuracy_{args.arch}.txt'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, train_accuracy)))

    file_name = f'TrainResults_TinyImagenet/test_tinyimagenet_accuracy_{args.arch}.txt'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, test_accuracy)))

    logger.info("Results saved to TrainResults_TinyImagenet/")
