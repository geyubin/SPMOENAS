import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from Operation import *
import math


def _decode_cell_segment(segment, steps=5):
    if len(segment) < steps * 5:
        raise ValueError(f"segment length must be >= {steps * 5}, got {len(segment)}")
    cell_geno = []
    offset = 0
    used = set()
    for i in range(steps):
        raw_n1, raw_n2, raw_op1, raw_op2, raw_comb = segment[offset:offset + 5]
        offset += 5
        # 越界保护，避免出现环
        max_idx = 2 + i
        n1 = int(raw_n1) % max_idx
        n2 = int(raw_n2) % max_idx
        # 寻找未使用的节点作为cell的最终输出，input0和1的索引为0和1
        if n1 >= 2:
            used.add(n1)
        if n2 >= 2:
            used.add(n2)
        op1_name = Operations_name[int(raw_op1) % len(Operations_name)]
        op2_name = Operations_name[int(raw_op2) % len(Operations_name)]
        comb_name = "concat" if (int(raw_comb) % 2) == 1 else "add"  # 0=add, 1=concat
        cell_geno.append((n1, n2, op1_name, op2_name, comb_name))

    all_nodes = list(range(2, 2 + steps))
    cell_concat = [idx for idx in all_nodes if idx not in used]
    if not cell_concat:
        cell_concat = [all_nodes[-1]]
    return cell_geno, cell_concat


def decode_network_from_list(inner_genotype):
    if len(inner_genotype) < 98:
        raise ValueError(f"inner_genotype length must be >= 98, got {len(inner_genotype)}")

    # The first three genes encode extra repetitions; each normal block template
    # must appear at least once in the network.
    stack_nums = [
        math.ceil(inner_genotype[0]) + 1,
        math.ceil(inner_genotype[1]) + 1,
        math.ceil(inner_genotype[2]) + 1,
    ]

    normal_raw = inner_genotype[3:78]
    reduce_raw = inner_genotype[78:98]

    if len(normal_raw) != 75 or len(reduce_raw) != 20:
        raise ValueError("inner_genotype must have 75 normal and 20 reduce genes after first 3 entries")

    normal_cells = []
    for j in range(3):
        seg = normal_raw[j * 25:(j + 1) * 25]
        geno, concat = _decode_cell_segment(seg, steps=5)
        normal_cells.append(SimpleNamespace(normal=geno, normal_concat=concat))

    reduce_cells = []
    for j in range(2):
        seg = reduce_raw[j * 10:(j + 1) * 10]
        nodes = []
        for i in range(5):
            src_idx = int(seg[2 * i])
            op_id = int(seg[2 * i + 1])
            nodes.append((src_idx, op_id))
        reduce_concat = list(range(2, 2 + 5))
        reduce_cells.append(SimpleNamespace(reduce=nodes, reduce_concat=reduce_concat))

    return SimpleNamespace(
        stack_nums=stack_nums,
        normal_cells=normal_cells,
        reduce_cells=reduce_cells,
    )


def make_reduce_pool_op(op_id):
    k = int(op_id) % 6
    if k == 0:
        return nn.MaxPool2d(2, stride=2, padding=0)
    if k == 1:
        return nn.AvgPool2d(2, stride=2, padding=0, count_include_pad=False)
    if k == 2:
        return nn.MaxPool2d(3, stride=2, padding=1)
    if k == 3:
        return nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    if k == 4:
        return nn.MaxPool2d(5, stride=2, padding=2)
    return nn.AvgPool2d(5, stride=2, padding=2, count_include_pad=False)


def drop_path(x, drop_prob=0.0, training=False):
    drop_prob = float(drop_prob)
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    if keep_prob <= 0.0:
        return torch.zeros_like(x)
    mask_shape = (x.size(0),) + (1,) * (x.dim() - 1)
    mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * mask


class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, steps=4):
        super(Cell, self).__init__()
        self.steps = steps
        self.C = C
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            self.geno = genotype.reduce
            self.concat = genotype.reduce_concat
        else:
            self.geno = genotype.normal
            self.concat = genotype.normal_concat
        self.compiler(C, reduction)

        self.multiplier = len(self.concat)

    def compiler(self, C, reduction):
        self.nodes = []
        self.ops = nn.ModuleList()
        self.combs = nn.ModuleList()
        if reduction and len(self.geno) > 0 and len(self.geno[0]) == 2:
            for i, (src_idx, op_id) in enumerate(self.geno):
                n = int(src_idx) % 2
                self.nodes.append(n)
                op = make_reduce_pool_op(op_id)
                self.ops.append(op)
        else:
            for (n1, n2, op1_name, op2_name, comb_name) in self.geno:
                self.nodes.append(n1)
                self.nodes.append(n2)

                stride1 = 2 if reduction and n1 < 2 else 1
                op1 = Operations[op1_name](C, stride1, False)
                if 'pool' in op1_name:
                    op1 = nn.Sequential(op1, nn.BatchNorm2d(C, affine=False))

                stride2 = 2 if reduction and n2 < 2 else 1
                op2 = Operations[op2_name](C, stride2, False)
                if 'pool' in op2_name:
                    op2 = nn.Sequential(op2, nn.BatchNorm2d(C, affine=False))

                self.ops.append(op1)
                self.ops.append(op2)

                if comb_name == 'add':
                    self.combs.append(None)
                else:
                    self.combs.append(ReLUConvBN(self.C * 2, self.C, 1, 1, 0))

    def forward(self, s0, s1, drop_path_prob=0.0):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        is_single = self.reduction and len(self.geno) > 0 and len(self.geno[0]) == 2
        if is_single:
            for i, op in enumerate(self.ops):
                idx = self.nodes[i]
                h = states[idx]
                s = op(h)
                if not isinstance(op, Identity):
                    s = drop_path(s, drop_path_prob, self.training)
                states.append(s)
        else:
            for i in range(self.steps):
                h1 = states[self.nodes[2 * i]]
                h2 = states[self.nodes[2 * i + 1]]
                op1 = self.ops[2 * i]
                op2 = self.ops[2 * i + 1]
                h1 = op1(h1)
                h2 = op2(h2)
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_path_prob, self.training)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_path_prob, self.training)
                comb = self.combs[i]
                if comb == None:
                    s = h1 + h2
                else:
                    s = torch.cat([h1, h2], dim=1)
                    s = comb(s)
                states += [s]

        return torch.cat([states[i] for i in self.concat], dim=1)


class NetworkCifar(nn.Module):
    def __init__(self, inner_genotype, num_classes=10, C=64, stem_multiplier=2,
                 use_aux=False, aux_head_type="cifar", dropout_rate=0.0,
                 stem_type="cifar", drop_path_prob=0.0):
        super(NetworkCifar, self).__init__()

        decoded = decode_network_from_list(inner_genotype)
        self.stack_nums = decoded.stack_nums
        self.normal_cells = decoded.normal_cells
        self.reduce_cells = decoded.reduce_cells
        self.stem_type = stem_type.lower()
        self.drop_path_prob = float(drop_path_prob)

        C_curr = stem_multiplier * C
        if self.stem_type in ("cifar", "tiny", "small"):
            self.stem = nn.Sequential(
                nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
                nn.ReLU(inplace=False)
            )
        elif self.stem_type in ("image", "imagenet", "imagenet2012"):
            self.stem = nn.Sequential(
                nn.Conv2d(3, C_curr, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(C_curr),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            raise ValueError(f"Unknown stem_type: {stem_type}")

        C_prev_prev, C_prev = C_curr, C_curr
        C_cell = C
        self.cells = nn.ModuleList()
        reduction_prev = False

        # stage 1: normal cell 1 repeated stack_nums[0] times
        for _ in range(self.stack_nums[0]):
            cell = Cell(self.normal_cells[0], C_prev_prev, C_prev, C_cell, False, reduction_prev, steps=5)
            self.cells.append(cell)
            reduction_prev = False
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_cell

        # reduction cell 1
        C_cell *= 2
        cell = Cell(self.reduce_cells[0], C_prev_prev, C_prev, C_cell, True, reduction_prev, steps=5)
        self.cells.append(cell)
        reduction_prev = True
        C_prev_prev, C_prev = C_prev, cell.multiplier * C_cell

        # stage 2: normal cell 2
        for _ in range(self.stack_nums[1]):
            cell = Cell(self.normal_cells[1], C_prev_prev, C_prev, C_cell, False, reduction_prev, steps=5)
            self.cells.append(cell)
            reduction_prev = False
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_cell

        # reduction cell 2
        C_cell *= 2
        cell = Cell(self.reduce_cells[1], C_prev_prev, C_prev, C_cell, True, reduction_prev, steps=5)
        self.cells.append(cell)
        reduction_prev = True
        C_prev_prev, C_prev = C_prev, cell.multiplier * C_cell

        # stage 3: normal cell 3
        for _ in range(self.stack_nums[2]):
            cell = Cell(self.normal_cells[2], C_prev_prev, C_prev, C_cell, False, reduction_prev, steps=5)
            self.cells.append(cell)
            reduction_prev = False
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_cell

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=float(dropout_rate)) if float(dropout_rate) > 0.0 else nn.Identity()
        self.classifier = nn.Linear(C_prev, num_classes)
        self.use_aux = use_aux
        self.aux_head_type = aux_head_type
        self.aux_head = self._build_aux_head(C_prev, num_classes, aux_head_type) if use_aux else None

    @staticmethod
    def _build_aux_head(C_in, num_classes, aux_head_type):
        aux_head_type = aux_head_type.lower()
        if aux_head_type == "cifar":
            return AuxHeadCIFAR(C_in, num_classes)
        if aux_head_type in ("image", "imagenet", "tiny", "tiny_imagenet"):
            return AuxHeadImage(C_in, num_classes)
        raise ValueError(f"Unknown aux_head_type: {aux_head_type}")

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        aux_logits = None
        if self.use_aux:
            aux_logits = self.aux_head(s1)
        out = self.global_pooling(s1)
        out = self.dropout(out.view(out.size(0), -1))
        logits = self.classifier(out)
        if self.use_aux:
            return logits, aux_logits
        return logits
