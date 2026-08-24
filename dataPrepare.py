import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import ConcatDataset, Subset, random_split
from autoaugment import CIFAR10Policy, ImageNetPolicy
import torchvision.datasets as datasets


IMAGENET2012_MEAN = (0.485, 0.456, 0.406)
IMAGENET2012_STD = (0.229, 0.224, 0.225)


class Cutout(object):
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def data_transform_cifar10():
    # 训练集图像预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),  # 图像切割
        transforms.RandomHorizontalFlip(),  # 水平反转
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),  # 通过标准化实现白化
        Cutout(16)
    ])

    # 验证集图像处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    ])

    return train_transform, test_transform


def data_search_transform_cifar10():
    # 训练集图像预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),  # 图像切割
        transforms.RandomHorizontalFlip(),  # 水平反转
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),  # 通过标准化实现白化
    ])

    # 验证集图像处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    ])

    return train_transform, test_transform


def data_transform_cinic10():
    cinic_mean = (0.47889522, 0.47227842, 0.43047404)
    cinic_std = (0.24205776, 0.23828046, 0.25874835)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
        Cutout(16)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
    ])

    return train_transform, test_transform


def build_cifar10_train_valid_test_loader(batch_size):
    train_transform, valid_transform = data_transform_cifar10()
    train_set = torchvision.datasets.CIFAR10(root=r'/data/job2/data/', train=True, download=True, transform=train_transform)
    num_workers = int(os.environ.get('NUM_WORKERS', 12))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(root=r'/data/job2/data/', train=False, download=True, transform=valid_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                              pin_memory=True)

    return train_loader, test_loader


def build_cinic10_train_valid_test_loader(batch_size, root='./data/cinic10', include_valid_in_train=False):
    train_transform, valid_transform = data_transform_cinic10()
    num_workers = int(os.environ.get('NUM_WORKERS', 12))
    train_path = os.path.join(root, 'train')
    valid_path = os.path.join(root, 'valid')
    test_path = os.path.join(root, 'test')

    train_set = torchvision.datasets.ImageFolder(train_path, transform=train_transform)
    if include_valid_in_train:
        valid_train_set = torchvision.datasets.ImageFolder(valid_path, transform=train_transform)
        train_set = ConcatDataset([train_set, valid_train_set])
    valid_set = torchvision.datasets.ImageFolder(valid_path, transform=valid_transform)
    test_set = torchvision.datasets.ImageFolder(test_path, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader


def build_search_cifar10_train_valid_test_loader(batch_size):
    train_transform, valid_transform = data_search_transform_cifar10()
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    num_workers = int(os.environ.get('NUM_WORKERS', 12))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16,
                                               pin_memory=True)

    return train_loader, valid_loader


def data_transform_cifar100():
    # 训练集图像预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    # 验证集图像处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    return train_transform, test_transform


def data_train_transform_cifar100():
    # 训练集图像预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop((32, 32), padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        Cutout(16)
    ])

    # 验证集图像处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    return train_transform, test_transform


def build_search_cifar100_train_valid_test_loader(batch_size):
    train_transform, valid_transform = data_transform_cifar100()
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16,
                                               pin_memory=True)

    return train_loader, valid_loader


def build_cifar100_train_valid_test_loader(batch_size):
    train_transform, valid_transform = data_train_transform_cifar100()
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12,
                                               pin_memory=True)

    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=valid_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=12,
                                              pin_memory=True)

    return train_loader, test_loader


def data_transform_cinic10():
    cinic_mean = (0.47889522, 0.47227842, 0.43047404)
    cinic_std = (0.24205776, 0.23828046, 0.25874835)
    # cinic_mean = (0.47889522, 0.47227842, 0.43047366)
    # cinic_std = (0.24205776, 0.23828046, 0.2592109)
    use_cutout = os.environ.get('USE_CUTOUT', '1') == '1'
    train_transforms = [
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
    ]
    if use_cutout:
        train_transforms.append(Cutout(16))
    train_transform = transforms.Compose(train_transforms)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
    ])

    return train_transform, test_transform


def data_search_transform_cinic10():
    # cinic_mean = (0.47889522, 0.47227842, 0.43047404)
    # cinic_std = (0.24205776, 0.23828046, 0.25874835)
    cinic_mean = (0.47889522, 0.47227842, 0.43047366)
    cinic_std = (0.24205776, 0.23828046, 0.2592109)
    use_cutout = os.environ.get('USE_CUTOUT', '0') == '1'
    train_transforms = [
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
    ]
    if use_cutout:
        train_transforms.append(Cutout(16))
    train_transform = transforms.Compose(train_transforms)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
    ])

    return train_transform, test_transform


def build_train_cinic10_train_valid_test_loader(batch_size, root='./data/cinic10'):
    train_transform, valid_transform = data_transform_cinic10()
    num_workers = int(os.environ.get('NUM_WORKERS', 16))
    train_path = os.path.join(root, 'train')
    valid_path = os.path.join(root, 'valid')
    test_path = os.path.join(root, 'test')

    train_set = torchvision.datasets.ImageFolder(train_path, transform=train_transform)
    valid_set = torchvision.datasets.ImageFolder(valid_path, transform=valid_transform)
    test_set = torchvision.datasets.ImageFolder(test_path, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader


def build_search_cinic10_train_valid_test_loader(batch_size, root='./data/cinic10'):
    train_transform, valid_transform = data_search_transform_cinic10()
    num_workers = int(os.environ.get('NUM_WORKERS', 16))
    train_path = os.path.join(root, 'train')
    valid_path = os.path.join(root, 'valid')

    train_set = torchvision.datasets.ImageFolder(train_path, transform=train_transform)
    valid_set = torchvision.datasets.ImageFolder(valid_path, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader


def build_search_tiny_imagenet_train_val_loader(root, batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        # ImageNetPolicy(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ])

    num_workers = int(os.environ.get('NUM_WORKERS', 16))
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def _imagenet_val_resize(image_size):
    return int(round(float(image_size) * 256.0 / 224.0))


def _require_imagefolder_dir(path, split_name):
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"ImageNet2012 {split_name} directory not found: {path}. "
            "Expected ImageFolder layout like root/train/n01440764/*.JPEG "
            "and root/val/n01440764/*.JPEG."
        )


def _validate_probability(name, value):
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _validate_ratio(name, value):
    value = float(value)
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def _subsample_indices(indices, ratio, name):
    ratio = _validate_ratio(name, ratio)
    if ratio >= 1.0:
        return indices
    subset_size = max(1, int(len(indices) * ratio))
    return indices[:subset_size]


def _build_imagenet_augment(augment_policy):
    augment_policy = (augment_policy or "none").lower()
    if augment_policy in ("none", "off", "false", "0"):
        return None
    if augment_policy in ("autoaugment", "auto"):
        return ImageNetPolicy()
    if augment_policy in ("randaugment", "rand"):
        if not hasattr(transforms, "RandAugment"):
            raise RuntimeError("torchvision.transforms.RandAugment is not available in this torchvision version.")
        return transforms.RandAugment()
    if augment_policy in ("trivialaugment", "trivial", "trivialaugmentwide"):
        if not hasattr(transforms, "TrivialAugmentWide"):
            raise RuntimeError("torchvision.transforms.TrivialAugmentWide is not available in this torchvision version.")
        return transforms.TrivialAugmentWide()
    if augment_policy == "augmix":
        if not hasattr(transforms, "AugMix"):
            raise RuntimeError("torchvision.transforms.AugMix is not available in this torchvision version.")
        return transforms.AugMix()
    raise ValueError(
        f"Unknown ImageNet augmentation policy: {augment_policy}. "
        "Expected one of: none, autoaugment, randaugment, trivialaugment, augmix."
    )


def _append_random_erasing(train_transforms, random_erasing_prob):
    random_erasing_prob = _validate_probability("random_erasing_prob", random_erasing_prob)
    if random_erasing_prob > 0.0:
        train_transforms.append(transforms.RandomErasing(p=random_erasing_prob))


def data_search_transform_imagenet2012(image_size=224, augment_policy="autoaugment",
                                       random_erasing_prob=0.25):
    normalize = transforms.Normalize(mean=IMAGENET2012_MEAN, std=IMAGENET2012_STD)
    train_transforms = [
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ]
    augment = _build_imagenet_augment(augment_policy)
    if augment is not None:
        train_transforms.append(augment)
    train_transforms.extend([
        transforms.ToTensor(),
        normalize,
    ])
    _append_random_erasing(train_transforms, random_erasing_prob)

    val_transform = transforms.Compose([
        transforms.Resize(_imagenet_val_resize(image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    return transforms.Compose(train_transforms), val_transform


def data_train_transform_imagenet2012(image_size=224, augment_policy=None, use_autoaugment=True,
                                      use_cutout=False, cutout_length=56, random_erasing_prob=0.25):
    normalize = transforms.Normalize(mean=IMAGENET2012_MEAN, std=IMAGENET2012_STD)
    if augment_policy is None:
        augment_policy = "autoaugment" if use_autoaugment else "none"
    train_transforms = [
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ]
    augment = _build_imagenet_augment(augment_policy)
    if augment is not None:
        train_transforms.append(augment)
    train_transforms.extend([
        transforms.ToTensor(),
        normalize,
    ])
    _append_random_erasing(train_transforms, random_erasing_prob)
    if use_cutout:
        train_transforms.append(Cutout(cutout_length))

    val_transform = transforms.Compose([
        transforms.Resize(_imagenet_val_resize(image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    return transforms.Compose(train_transforms), val_transform


def build_search_imagenet2012_train_val_loader(root, batch_size, image_size=224,
                                               augment_policy="autoaugment",
                                               random_erasing_prob=0.25,
                                               valid_ratio=0.2,
                                               split_seed=2025,
                                               train_subset_ratio=1.0,
                                               valid_subset_ratio=1.0):
    valid_ratio = float(valid_ratio)
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError(f"valid_ratio must be in (0, 1), got {valid_ratio}")
    train_transform, val_transform = data_search_transform_imagenet2012(
        image_size,
        augment_policy=augment_policy,
        random_erasing_prob=random_erasing_prob,
    )
    num_workers = int(os.environ.get('NUM_WORKERS', 16))
    train_dir = os.path.join(root, 'train')
    _require_imagefolder_dir(train_dir, 'train')

    train_all = datasets.ImageFolder(train_dir, transform=train_transform)
    val_all = datasets.ImageFolder(train_dir, transform=val_transform)
    total_size = len(train_all)
    val_size = max(1, int(total_size * valid_ratio))
    train_size = total_size - val_size
    if train_size <= 0:
        raise ValueError(f"ImageNet2012 train split is empty after valid_ratio={valid_ratio}")

    rng = np.random.default_rng(int(split_seed))
    indices = rng.permutation(total_size)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    train_indices = _subsample_indices(train_indices, train_subset_ratio, "train_subset_ratio")
    val_indices = _subsample_indices(val_indices, valid_subset_ratio, "valid_subset_ratio")

    train_set = Subset(train_all, train_indices)
    val_set = Subset(val_all, val_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def build_train_imagenet2012_train_val_loader(root, batch_size, image_size=224,
                                              augment_policy=None, use_autoaugment=True,
                                              use_cutout=False, random_erasing_prob=0.25):
    train_transform, val_transform = data_train_transform_imagenet2012(
        image_size=image_size,
        augment_policy=augment_policy,
        use_autoaugment=use_autoaugment,
        use_cutout=use_cutout,
        random_erasing_prob=random_erasing_prob,
    )
    num_workers = int(os.environ.get('NUM_WORKERS', 16))
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')
    _require_imagefolder_dir(train_dir, 'train')
    _require_imagefolder_dir(val_dir, 'val')

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def build_train_tiny_imagenet_train_val_loader(root, batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.7, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        ImageNetPolicy(),
        transforms.ToTensor(),
        normalize,
        Cutout(16)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    num_workers = int(os.environ.get('NUM_WORKERS', 16))
    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'test')

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    test_set = datasets.ImageFolder(test_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def build_train_Optimizer_Loss(model, momentum=0.9, lr_max=0.025, l2_reg=5e-4, epochs=400,
                               device=None, label_smoothing=0.0, warmup_epochs=0):
    device1 = device
    # model.to(device1)
    try:
        train_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device1)
        eval_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device1)
    except TypeError:
        train_criterion = nn.CrossEntropyLoss().to(device1)
        eval_criterion = nn.CrossEntropyLoss().to(device1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr_max,
        momentum=momentum,
        weight_decay=l2_reg,
        nesterov=True
    )

    warmup_epochs = max(0, min(int(warmup_epochs), int(epochs)))
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / warmup_epochs,
            total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    return train_criterion, eval_criterion, optimizer, scheduler
