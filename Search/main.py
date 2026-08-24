import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Mopso import Mopso
from dataPrepare import (
    build_search_cifar10_train_valid_test_loader,
    build_search_cifar100_train_valid_test_loader,
    build_search_cinic10_train_valid_test_loader,
    build_search_tiny_imagenet_train_val_loader,
)


def set_seed(seed=0):
    if seed is None or int(seed) < 0:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed for particle initialization, subset sampling, and model training. Use -1 to disable fixed seeding.")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "cinic10", "tiny"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_channels", type=int, default=64)
    parser.add_argument("--model_stem_type", type=str, default="auto",
                        choices=["auto", "cifar", "tiny", "image"])
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=25)
    parser.add_argument("--mesh_div", type=int, default=10)
    parser.add_argument("--thresh", type=int, default=20)
    parser.add_argument("--subset_ratio", type=float, default=0.5)
    parser.add_argument("--teacher_num", type=int, default=5)
    parser.add_argument("--teacher_epochs", type=int, default=50)
    parser.add_argument("--lambda_kd", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=1.0)
    return parser.parse_args()


def validate_ratio(name, value):
    if value is None:
        return None
    if not (0 < value <= 1):
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def resolve_device(cuda_index):
    if cuda_index is None or cuda_index < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{cuda_index}")


def build_search_loaders(dataset, batch_size):
    if dataset == "cifar10":
        train_loader, valid_loader = build_search_cifar10_train_valid_test_loader(batch_size)
        return (train_loader, valid_loader), 10
    if dataset == "cifar100":
        train_loader, valid_loader = build_search_cifar100_train_valid_test_loader(batch_size)
        return (train_loader, valid_loader), 100
    if dataset == "cinic10":
        train_loader, valid_loader = build_search_cinic10_train_valid_test_loader(batch_size)
        return (train_loader, valid_loader), 10
    tiny_root = os.environ.get("TINY_ROOT", "./data/tiny-imagenet-200")
    train_loader, valid_loader = build_search_tiny_imagenet_train_val_loader(tiny_root, batch_size)
    return (train_loader, valid_loader), 200


def resolve_model_stem_type(model_stem_type):
    if model_stem_type != "auto":
        return model_stem_type
    return "cifar"


def format_array_full(array, precision=8):
    return np.array2string(
        np.asarray(array),
        precision=precision,
        threshold=np.inf,
        max_line_width=np.inf,
        suppress_small=False,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    args.subset_ratio = validate_ratio("subset_ratio", args.subset_ratio)

    w = 0.729
    c1 = 1.46
    c2 = 1.46
    min_ = np.zeros(98)
    max_ = np.array([3, 3, 3,
                     1, 1, 7, 7, 1, 2, 2, 7, 7, 1, 3, 3, 7, 7, 1, 4, 4, 7, 7, 1, 5, 5, 7, 7, 1,
                     1, 1, 7, 7, 1, 2, 2, 7, 7, 1, 3, 3, 7, 7, 1, 4, 4, 7, 7, 1, 5, 5, 7, 7, 1,
                     1, 1, 7, 7, 1, 2, 2, 7, 7, 1, 3, 3, 7, 7, 1, 4, 4, 7, 7, 1, 5, 5, 7, 7, 1,
                     1, 4, 1, 4, 1, 4, 1, 4, 1, 4,
                     1, 4, 1, 4, 1, 4, 1, 4, 1, 4])

    (train_loader, valid_loader), num_classes = build_search_loaders(
        args.dataset,
        args.batch_size,
    )
    device = resolve_device(args.cuda)
    model_stem_type = resolve_model_stem_type(args.model_stem_type)

    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print(f"dataset: {args.dataset}")
    print(f"batch_size: {args.batch_size}")
    print(f"train_samples: {len(train_loader.dataset)}")
    print(f"valid_samples: {len(valid_loader.dataset)}")
    print(f"model_channels: {args.model_channels}")
    print(f"model_stem_type: {model_stem_type}")
    print(f"particles: {args.particles}")
    print(f"cycles: {args.cycles}")
    print(f"teacher_num: {args.teacher_num}")
    print(f"teacher_epochs: {args.teacher_epochs}")
    print(f"subset_ratio: {args.subset_ratio}")
    print(f"lambda_kd: {args.lambda_kd}")

    start_time = time.time()
    mopso_ = Mopso(
        args.particles,
        w,
        c1,
        c2,
        max_,
        min_,
        args.thresh,
        train_loader,
        valid_loader,
        args.mesh_div,
        device,
        teacher_num=args.teacher_num,
        teacher_epochs=args.teacher_epochs,
        subset_ratio=args.subset_ratio,
        num_classes=num_classes,
        lambda_kd=args.lambda_kd,
        T=args.T,
        model_channels=args.model_channels,
        model_stem_type=model_stem_type,
    )
    pareto_in, pareto_fitness = mopso_.done(args.cycles)
    elapsed_time = time.time() - start_time

    print("pareto positions:")
    print(format_array_full(pareto_in))
    print("pareto fitness:")
    print(format_array_full(pareto_fitness))

    teacher_pretrain_elapsed_time = getattr(mopso_, "teacher_pretrain_elapsed_time", 0.0)
    print(f"total_gpu_days: {elapsed_time / 86400:.4f} GPU days")
    print(f"teacher_pretrain_gpu_days: {teacher_pretrain_elapsed_time / 86400:.4f} GPU days")

    result_dir = f"job2_SearchResults_{args.dataset}"
    os.makedirs(result_dir, exist_ok=True)
    ratio_tag = str(args.subset_ratio).replace(".", "_")
    seed_tag = str(args.seed) if args.seed is not None and args.seed >= 0 else "random"

    file = os.path.join(result_dir, f"PF_position_ratio_{ratio_tag}_seed_{seed_tag}.txt")
    np.savetxt(file, np.asarray(pareto_in, dtype=float), fmt="%.10f")

    file = os.path.join(result_dir, f"PF_fitness_ratio_{ratio_tag}_seed_{seed_tag}.txt")
    np.savetxt(file, np.asarray(pareto_fitness, dtype=float), fmt="%.10f")


if __name__ == "__main__":
    main()
