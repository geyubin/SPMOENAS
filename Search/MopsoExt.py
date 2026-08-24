import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import time
import gc
import random
import os

import update
from SearchSpace import NetworkCifar
from dataPrepare import build_train_Optimizer_Loss


class DeterministicIndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, base_seed=2025, return_index=False):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.base_seed = int(base_seed)
        self.return_index = bool(return_index)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        base_index = self.indices[int(position)]
        seed = (self.base_seed + base_index) % (2 ** 32 - 1)

        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            sample = self.dataset[base_index]
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)

        if not self.return_index:
            return sample
        if isinstance(sample, tuple):
            return (*sample, int(position))
        return sample, int(position)


class CachedTensorIndexedDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, return_index=False):
        self.inputs = inputs
        self.targets = targets
        self.return_index = bool(return_index)

    def __len__(self):
        return int(self.inputs.size(0))

    def __getitem__(self, position):
        position = int(position)
        if self.return_index:
            return self.inputs[position], self.targets[position], position
        return self.inputs[position], self.targets[position]


class MopsoExtMixin:

    def _clear_runtime_cache(self):
        gc.collect()
        if isinstance(self.device, torch.device) and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _network_kwargs(self):
        return {
            "num_classes": getattr(self, "num_classes", 10),
            "C": getattr(self, "model_channels", 64),
            "stem_type": getattr(self, "model_stem_type", "cifar"),
            "use_aux": getattr(self, "search_use_aux", False),
            "aux_head_type": getattr(self, "search_aux_head_type", "cifar"),
            "drop_path_prob": 0.0,
        }

    def _split_logits(self, outputs):
        if isinstance(outputs, tuple):
            return outputs[0], outputs[1]
        return outputs, None

    def _loss_with_aux(self, criterion, outputs, targets):
        logits, aux_logits = self._split_logits(outputs)
        loss = criterion(logits, targets)
        if aux_logits is not None:
            loss = loss + getattr(self, "search_aux_weight", 0.4) * criterion(aux_logits, targets)
        return logits, loss

    def _unique_cell_indices(self, model):
        if not hasattr(model, "stack_nums"):
            total = len(model.cells)
            return [total - 1] if total > 0 else []

        s0, s1, s2 = model.stack_nums
        total = len(model.cells)
        indices = []

        if s0 > 0:
            idx0 = s0 - 1
            if 0 <= idx0 < total:
                indices.append(idx0)

        if s1 > 0:
            idx1 = s0 + s1
            if 0 <= idx1 < total:
                indices.append(idx1)

        last_idx = total - 1
        if last_idx >= 0 and last_idx not in indices:
            indices.append(last_idx)

        return indices

    # 获取3个normal block首次出现的位置索引
    def _normal_template_indices(self, model):
        if not hasattr(model, "stack_nums"):
            return []
        s0, s1, s2 = model.stack_nums
        total = len(model.cells)
        indices = []
        if s0 > 0 and total > 0:
            indices.append(0)
        idx1 = s0 + 1
        if s1 > 0 and 0 <= idx1 < total:
            indices.append(idx1)
        idx2 = s0 + 1 + s1 + 1
        if s2 > 0 and 0 <= idx2 < total:
            indices.append(idx2)
        return indices

    def _extract_block_features(self, model):
        normal_idx = self._normal_template_indices(model)
        cell_feats = []
        for cell_idx in normal_idx:
            if cell_idx < 0 or cell_idx >= len(model.cells):
                continue
            cell = model.cells[cell_idx]
            if not hasattr(cell, "geno") or not hasattr(cell, "nodes") or not hasattr(cell, "ops"):
                continue
            op_feats = []
            state_depths = [0.0, 0.0]
            steps = getattr(cell, "steps", 0)
            for i in range(steps):
                if 2 * i + 1 >= len(cell.nodes) or 2 * i + 1 >= len(cell.ops):
                    break
                src1 = cell.nodes[2 * i]
                src2 = cell.nodes[2 * i + 1]
                if src1 < 0 or src1 >= len(state_depths) or src2 < 0 or src2 >= len(state_depths):
                    depth_s = max(state_depths)
                else:
                    depth_s = max(state_depths[src1], state_depths[src2]) + 1.0
                state_depths.append(depth_s)
                op1 = cell.ops[2 * i]
                op2 = cell.ops[2 * i + 1]
                for op in (op1, op2):
                    op_type = 0
                    cin = 0
                    cout = 0
                    k = 0
                    for module in op.modules():
                        if isinstance(module, nn.Conv2d):
                            op_type = 1
                            cin = int(module.in_channels)
                            cout = int(module.out_channels)
                            if isinstance(module.kernel_size, tuple):
                                k = int(module.kernel_size[0])
                            else:
                                k = int(module.kernel_size)
                            break
                    op_feats.append((op_type, cin, cout, k, depth_s))
            cell_feats.append(op_feats)
        return normal_idx, cell_feats

    def _split_normal_cell_codes(self, arch_disc):
        arr = np.asarray(arch_disc, dtype=float)
        if arr.size < 78:
            return []
        normal_raw = arr[3:78]
        if normal_raw.size != 75:
            return []
        cells = []
        for j in range(3):
            seg = normal_raw[j * 25:(j + 1) * 25]
            cells.append(seg.copy())
        return cells  # 长度为 3 的列表，每个元素是对应 normal cell 的 25 维编码向量

    def _shared_search_subset_ratio(self):
        teacher_ratio = float(getattr(self, "teacher_subset_ratio", 1.0))
        student_ratio = float(getattr(self, "student_subset_ratio", teacher_ratio))
        if abs(teacher_ratio - student_ratio) > 1e-12 and not getattr(self, "_warned_shared_subset_ratio", False):
            print(
                "[KD cache] teacher_subset_ratio and student_subset_ratio differ; "
                f"using student_subset_ratio={student_ratio} for the shared KD training subset."
            )
            self._warned_shared_subset_ratio = True
        return student_ratio

    def _get_shared_train_indices(self):
        base_loader = self.input_train
        total_size = len(base_loader.dataset)
        ratio = self._shared_search_subset_ratio()
        subset_size = total_size if ratio >= 1.0 else max(1, int(total_size * float(ratio)))
        if (
            not hasattr(self, "_shared_train_indices")
            or self._shared_train_indices is None
            or len(self._shared_train_indices) != subset_size
        ):
            if subset_size >= total_size:
                indices = np.arange(total_size, dtype=np.int64)
            else:
                indices = np.random.choice(total_size, subset_size, replace=False).astype(np.int64)
            self._shared_train_indices = indices
            self._shared_train_tensor_cache = None
            print(f"[KD cache] shared train subset size: {subset_size}/{total_size}")
        return self._shared_train_indices

    def _get_shared_train_tensor_cache(self):
        cache = getattr(self, "_shared_train_tensor_cache", None)
        indices = self._get_shared_train_indices()
        if cache is not None:
            cached_inputs, cached_targets = cache
            if int(cached_inputs.size(0)) == len(indices):
                return cache

        base_loader = self.input_train
        source_dataset = DeterministicIndexedSubset(
            base_loader.dataset,
            indices,
            base_seed=2025,
            return_index=True,
        )
        if len(source_dataset) == 0:
            return None

        sample = source_dataset[0]
        sample_input = torch.as_tensor(sample[0])
        sample_target = torch.as_tensor(sample[1])
        input_bytes = len(source_dataset) * sample_input.numel() * sample_input.element_size()
        target_bytes = len(source_dataset) * max(1, sample_target.numel()) * sample_target.element_size()
        total_gib = (input_bytes + target_bytes) / (1024 ** 3)
        max_gib = float(os.environ.get("KD_IMAGE_CACHE_MAX_GIB", "2.0"))
        if max_gib > 0.0 and total_gib > max_gib:
            print(
                f"[KD cache] shared train image cache skipped: estimated {total_gib:.3f} GiB "
                f"> KD_IMAGE_CACHE_MAX_GIB={max_gib:.3f} GiB"
            )
            return None

        t0 = time.time()
        inputs_cache = torch.empty(
            (len(source_dataset),) + tuple(sample_input.shape),
            dtype=sample_input.dtype,
        )
        targets_cache = torch.empty(
            (len(source_dataset),) + tuple(sample_target.shape),
            dtype=sample_target.dtype,
        )
        build_loader = DataLoader(
            source_dataset,
            batch_size=base_loader.batch_size,
            shuffle=False,
            num_workers=base_loader.num_workers,
            pin_memory=False,
        )
        for batch in build_loader:
            inputs, targets, positions = self._unpack_train_batch(batch)
            positions = positions.detach().cpu().long()
            inputs_cache.index_copy_(0, positions, inputs.detach().cpu())
            targets_cache.index_copy_(0, positions, torch.as_tensor(targets).detach().cpu())

        self._shared_train_tensor_cache = (inputs_cache, targets_cache)
        elapsed = time.time() - t0
        print(
            f"[KD cache] shared train image cache: {total_gib:.3f} GiB, "
            f"built in {elapsed:.3f} s"
        )
        return self._shared_train_tensor_cache

    def _build_shared_train_loader(self, shuffle=True, return_index=False):
        base_loader = self.input_train
        tensor_cache = self._get_shared_train_tensor_cache()
        if tensor_cache is not None:
            dataset = CachedTensorIndexedDataset(
                tensor_cache[0],
                tensor_cache[1],
                return_index=return_index,
            )
            num_workers = 0
        else:
            dataset = DeterministicIndexedSubset(
                base_loader.dataset,
                self._get_shared_train_indices(),
                base_seed=2025,
                return_index=return_index,
            )
            num_workers = base_loader.num_workers
        pin_memory = getattr(base_loader, "pin_memory", True)
        return DataLoader(
            dataset,
            batch_size=base_loader.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def _unpack_train_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            return batch[0], batch[1], batch[2]
        return batch[0], batch[1], None

    def _build_subset_loader(self):
        if getattr(self, "use_kd", False):
            return self._build_shared_train_loader(shuffle=True, return_index=False)

        base_loader = self.input_train
        dataset = base_loader.dataset
        total_size = len(dataset)
        subset_size = max(1, int(total_size * float(self.teacher_subset_ratio)))
        indices = np.random.choice(total_size, subset_size, replace=False)
        subset = Subset(dataset, indices.tolist())
        batch_size = base_loader.batch_size
        num_workers = base_loader.num_workers
        pin_memory = getattr(base_loader, "pin_memory", True)
        return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    def _train_teacher_model(self, arch_vector, train_loader):
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        arch_array = np.asarray(arch_vector, dtype=float)
        discrete_arch = update.vectorized_discretize(arch_array, self.min_, self.max_)
        model = NetworkCifar(discrete_arch.tolist(), **self._network_kwargs())
        num_classes = model.classifier.out_features
        unique_idx = self._unique_cell_indices(model)
        cell_heads = nn.ModuleList()
        for cell_idx in unique_idx:
            cell = model.cells[cell_idx]
            C_out = cell.multiplier * cell.C
            head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(C_out, num_classes)
            )
            cell_heads.append(head)
        model.to(self.device)
        cell_heads.to(self.device)
        train_criterion, eval_criterion, optimizer, scheduler = build_train_Optimizer_Loss(
            model, epochs=self.teacher_epochs, device=self.device
        )
        optimizer.add_param_group({"params": cell_heads.parameters()})
        for epoch in range(self.teacher_epochs):
            print("Epoch:", epoch)
            model.train()
            cell_heads.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_cell_outs = [None for _ in range(len(cell_heads))]
                hooks = []

                def make_hook(pos):
                    def hook(module, inp, out):
                        batch_cell_outs[pos] = out
                    return hook

                for pos, cell_idx in enumerate(unique_idx):
                    cell = model.cells[cell_idx]
                    hooks.append(cell.register_forward_hook(make_hook(pos)))

                optimizer.zero_grad()
                outputs = model(inputs)
                logits, loss_main = self._loss_with_aux(train_criterion, outputs, targets)
                loss_main.backward()

                cell_loss = 0.0
                for i, head in enumerate(cell_heads):
                    feat = batch_cell_outs[i]
                    if feat is None:
                        continue
                    cell_logits = head(feat.detach())
                    cell_loss = cell_loss + train_criterion(cell_logits, targets)
                if cell_loss != 0:
                    cell_loss.backward()
                optimizer.step()
                for h in hooks:
                    h.remove()
            scheduler.step()
        return model, cell_heads

    def _evaluate_teacher_accuracy(self, model, valid_loader):
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                logits, _ = self._split_logits(outputs)
                preds = logits.argmax(dim=1)
                total += targets.size(0)
                correct += (preds == targets).sum().item()
        return correct / max(1, total)

    def _extract_teacher_info(self, model, data_loader):
        """提取教师模型信息 - 优化版本，避免存储所有中间输出"""
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()

        # 只提取权重，不再存储所有中间输出（这是内存杀手！）
        cell_weights = []
        normal_idx = self._normal_template_indices(model)
        for cell_idx in normal_idx:
            cell = model.cells[cell_idx]
            w_list = []
            for module in cell.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    w_list.append(module.weight.detach().cpu().clone())
            cell_weights.append(w_list)

        # 返回空列表代替cell_outputs，因为我们不再预存所有中间输出
        # 中间输出将在需要时实时计算
        num_unique = len(self._unique_cell_indices(model))
        cell_outputs = [[] for _ in range(num_unique)]

        return cell_outputs, cell_weights

    def _forward_with_stage_features(self, model, inputs):
        s0 = s1 = model.stem(inputs)
        feats = []

        stack_nums = getattr(model, "stack_nums", [0, 0, 0])
        s0_n, s1_n, s2_n = stack_nums
        cells = model.cells
        idx = 0

        for _ in range(s0_n):
            if idx >= len(cells):
                break
            cell = cells[idx]
            idx += 1
            s0, s1 = s1, cell(s0, s1, getattr(model, "drop_path_prob", 0.0))

        feat_stage1 = s1
        if idx < len(cells):
            cell = cells[idx]
            idx += 1
            s0, s1 = s1, cell(s0, s1, getattr(model, "drop_path_prob", 0.0))

        for _ in range(s1_n):
            if idx >= len(cells):
                break
            cell = cells[idx]
            idx += 1
            s0, s1 = s1, cell(s0, s1, getattr(model, "drop_path_prob", 0.0))

        feat_stage2 = s1
        if idx < len(cells):
            cell = cells[idx]
            idx += 1
            s0, s1 = s1, cell(s0, s1, getattr(model, "drop_path_prob", 0.0))

        for _ in range(s2_n):
            if idx >= len(cells):
                break
            cell = cells[idx]
            idx += 1
            s0, s1 = s1, cell(s0, s1, getattr(model, "drop_path_prob", 0.0))

        feat_stage3 = s1
        aux_logits = model.aux_head(s1) if getattr(model, "use_aux", False) else None
        out = model.global_pooling(s1)
        logits = model.classifier(out.view(out.size(0), -1))

        feats.append(feat_stage1)
        feats.append(feat_stage2)
        feats.append(feat_stage3)
        return logits, feats, aux_logits

    def _has_teacher_stage_cache(self):
        cache = getattr(self, "teacher_stage_cache", None)
        return bool(cache) and len(cache) == len(getattr(self, "teacher_pool", []))

    def _build_teacher_stage_cache(self):
        if not getattr(self, "use_kd", False):
            return
        if not getattr(self, "teacher_pool", None):
            return

        t0 = time.time()
        cache_loader = self._build_shared_train_loader(shuffle=False, return_index=True)
        num_samples = len(cache_loader.dataset)
        teacher_stage_cache = []
        total_bytes = 0

        for teacher_idx, teacher_model in enumerate(self.teacher_pool):
            teacher_model.to(self.device)
            teacher_model.eval()
            stage_cache = None

            with torch.no_grad():
                for batch in cache_loader:
                    inputs, _, positions = self._unpack_train_batch(batch)
                    inputs = inputs.to(self.device)
                    positions = positions.detach().cpu().long()

                    _, t_stage_feats, _ = self._forward_with_stage_features(teacher_model, inputs)
                    t_stage_vecs = [
                        F.adaptive_avg_pool2d(t_feat, 1).view(t_feat.size(0), -1).detach().cpu().float()
                        for t_feat in t_stage_feats
                    ]

                    if stage_cache is None:
                        stage_cache = [
                            torch.empty((num_samples, vec.size(1)), dtype=torch.float32)
                            for vec in t_stage_vecs
                        ]

                    for stage_idx, vec in enumerate(t_stage_vecs):
                        stage_cache[stage_idx].index_copy_(0, positions, vec)

                    inputs = t_stage_feats = t_stage_vecs = None

            teacher_model.cpu()
            if stage_cache is None:
                stage_cache = []
            teacher_stage_cache.append(stage_cache)
            teacher_bytes = sum(stage_tensor.numel() * stage_tensor.element_size() for stage_tensor in stage_cache)
            total_bytes += teacher_bytes
            print(
                f"[KD cache] teacher {teacher_idx} stage cache: "
                f"{teacher_bytes / (1024 ** 3):.3f} GiB"
            )

        self.teacher_stage_cache = teacher_stage_cache
        elapsed = time.time() - t0
        print(
            f"[KD cache] built teacher stage cache in {elapsed:.3f} s, "
            f"total memory: {total_bytes / (1024 ** 3):.3f} GiB"
        )

    def _get_cached_teacher_stage_vectors(self, positions):
        positions = positions.detach().cpu().long()
        cached_all = []
        for stage_cache in getattr(self, "teacher_stage_cache", []):
            cached_all.append([
                stage_tensor.index_select(0, positions).to(self.device)
                for stage_tensor in stage_cache
            ])
        return cached_all

    def _get_cell_ops_and_features(self, cell):
        ops = []
        feats = []
        if not hasattr(cell, "nodes") or not hasattr(cell, "ops"):
            return ops, feats
        state_depths = [0.0, 0.0]
        steps = getattr(cell, "steps", 0)
        for i in range(steps):
            if 2 * i + 1 >= len(cell.nodes) or 2 * i + 1 >= len(cell.ops):
                break
            src1 = cell.nodes[2 * i]
            src2 = cell.nodes[2 * i + 1]
            if src1 < 0 or src1 >= len(state_depths) or src2 < 0 or src2 >= len(state_depths):
                depth_s = max(state_depths)
            else:
                depth_s = max(state_depths[src1], state_depths[src2]) + 1.0
            state_depths.append(depth_s)
            op1 = cell.ops[2 * i]
            op2 = cell.ops[2 * i + 1]
            for op in (op1, op2):
                op_type = 0
                cin = 0
                cout = 0
                k = 0
                for module in op.modules():
                    if isinstance(module, nn.Conv2d):
                        op_type = 1
                        cin = int(module.in_channels)
                        cout = int(module.out_channels)
                        if isinstance(module.kernel_size, tuple):
                            k = int(module.kernel_size[0])
                        else:
                            k = int(module.kernel_size)
                        break
                ops.append(op)
                feats.append((op_type, cin, cout, k, depth_s))
        return ops, feats

    # 返回与学生中每个操作节点结构最相似的教师中操作节点的索引
    def _match_ops_by_structure(self, t_feats, s_feats):
        eps = 1e-8
        indices = []
        for o_j, cin_j, cout_j, k_j, depth_j in s_feats:
            best_idx = None
            best_M = None
            for idx, (o_i, cin_i, cout_i, k_i, depth_i) in enumerate(t_feats):
                if not (o_i == 1 and o_j == 1):
                    M_ij = 0.0
                else:
                    s_op = 1.0
                    if k_i != k_j:
                        s_op = 0.5

                    def sim_attr(a, b):
                        a = float(a)
                        b = float(b)
                        if a <= 0 or b <= 0:
                            return 0.0
                        val = 1.0 - abs(a - b) / max(a, b)
                        if val < 0.0:
                            val = 0.0
                        return val

                    s_cin = sim_attr(cin_i, cin_j)
                    s_cout = sim_attr(cout_i, cout_j)
                    s_k = sim_attr(k_i, k_j)
                    s_depth = sim_attr(depth_i, depth_j)

                    components = [s_op, s_cin, s_cout, s_k, s_depth]
                    components = [max(c, eps) for c in components]
                    b_ij = float(len(components))
                    M_ij = (1.0 / b_ij) * sum(np.log(c) for c in components)

                if best_M is None or M_ij > best_M:
                    best_M = M_ij
                    best_idx = idx
            indices.append(best_idx)
        return indices

    def _transfer_conv_weight(self, student_conv, teacher_conv):
        with torch.no_grad():
            W_t = teacher_conv.weight.detach()
            W_s = student_conv.weight
            Cout_s, Cin_s, Ks_s, _ = W_s.shape
            Cout_t, Cin_t, Ks_t, _ = W_t.shape

            if Ks_t == Ks_s:
                W_resized = W_t
            elif Ks_t > Ks_s:
                start = (Ks_t - Ks_s) // 2
                end = start + Ks_s
                W_resized = W_t[:, :, start:end, start:end]
            else:
                pad = (Ks_s - Ks_t) // 2
                W_resized = W_t.new_zeros((Cout_t, Cin_t, Ks_s, Ks_s))
                W_resized[:, :, pad:pad + Ks_t, pad:pad + Ks_t] = W_t

            W_final = W_s.detach().clone()
            cout_min = min(Cout_s, Cout_t)
            cin_min = min(Cin_s, Cin_t)
            W_final[:cout_min, :cin_min, :, :] = W_resized[:cout_min, :cin_min, :, :]
            # - 不改变 W_s 这个张量本身的对象和形状；把 W_final 中的数值逐元素拷贝到 W_s 里面
            W_s.copy_(W_final)

    def _init_cell_from_teacher_cell(self, student_model, teacher_model, s_cell_idx, t_cell_idx):
        if s_cell_idx < 0 or s_cell_idx >= len(student_model.cells):
            return
        if t_cell_idx < 0 or t_cell_idx >= len(teacher_model.cells):
            return
        s_cell = student_model.cells[s_cell_idx]
        t_cell = teacher_model.cells[t_cell_idx]
        s_ops, s_feats = self._get_cell_ops_and_features(s_cell)
        t_ops, t_feats = self._get_cell_ops_and_features(t_cell)
        if not s_ops or not t_ops:
            return
        match_indices = self._match_ops_by_structure(t_feats, s_feats)  # 获取学生中每个操作节点结构最相似的教师中操作节点的索引
        for s_idx, t_idx in enumerate(match_indices):
            if t_idx is None:
                continue
            s_op = s_ops[s_idx]
            t_op = t_ops[t_idx]
            s_conv = None
            t_conv = None
            for module in s_op.modules():
                if isinstance(module, nn.Conv2d):
                    s_conv = module
                    break
            for module in t_op.modules():
                if isinstance(module, nn.Conv2d):
                    t_conv = module
                    break
            if s_conv is None or t_conv is None:
                continue
            self._transfer_conv_weight(s_conv, t_conv)

    def _initialize_teacher_pool(self):
        if self.particals <= 0:
            return
        if getattr(self, "teacher_num", 0) <= 0 or getattr(self, "teacher_epochs", 0) <= 0:
            return
        t0 = time.time()
        subset_loader = self._build_subset_loader()
        norm = (self.in_ - self.min_) / (self.max_ - self.min_ + 1e-8)
        scores = norm.mean(axis=1)
        order = np.argsort(scores)
        num_teachers = min(self.teacher_num, len(order))
        if num_teachers <= 0:
            return
        positions = np.linspace(0, len(order) - 1, num_teachers).astype(int)
        selected_idx = order[positions]
        self.teacher_pool = []
        self.teacher_arch = []
        self.teacher_valid_accs = []
        self.best_teacher_index = None
        self.teacher_pretrain_elapsed_time = 0.0
        for idx in selected_idx:
            arch_vec = self.in_[idx]
            train_t0 = time.time()
            model, cell_heads = self._train_teacher_model(arch_vec, subset_loader)
            train_elapsed = time.time() - train_t0
            self.teacher_pretrain_elapsed_time += train_elapsed
            print(f'complete teacher model for particle index {idx}')
            print(
                f'teacher pretrain time for particle index {idx}: {train_elapsed:.3f} s '
                f'({train_elapsed / 86400:.6f} GPU days)'
            )

            # 将模型移到CPU以节省GPU内存
            teacher_valid_acc = self._evaluate_teacher_accuracy(model, self.input_valid)
            self.teacher_valid_accs.append(teacher_valid_acc)
            print(f'teacher validation acc for particle index {idx}: {teacher_valid_acc:.6f}')

            model.cpu()
            cell_heads.cpu()

            self.teacher_pool.append(model)
            self.teacher_arch.append(update.vectorized_discretize(arch_vec, self.min_, self.max_))
            self.teacher_cell_heads.append(cell_heads)
            cell_out, cell_w = self._extract_teacher_info(model, subset_loader)
            self.teacher_cell_outputs.append(cell_out)
            self.teacher_cell_weights.append(cell_w)
            if not hasattr(self, "teacher_cell_indices"):
                self.teacher_cell_indices = []
            if not hasattr(self, "teacher_block_feats"):
                self.teacher_block_feats = []
            cell_indices, cell_feats = self._extract_block_features(model)
            self.teacher_cell_indices.append(cell_indices)
            self.teacher_block_feats.append(cell_feats)

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.teacher_valid_accs:
            self.best_teacher_index = int(np.argmax(np.asarray(self.teacher_valid_accs, dtype=float)))
            print(
                f'best teacher index: {self.best_teacher_index}, '
                f'valid_acc: {self.teacher_valid_accs[self.best_teacher_index]:.6f}'
            )

        self._build_teacher_stage_cache()

        t1 = time.time()
        elapsed = t1 - t0
        self.teacher_pool_elapsed_time = elapsed
        print(
            f'teacher pretrain total time: {self.teacher_pretrain_elapsed_time:.3f} s '
            f'({self.teacher_pretrain_elapsed_time / 86400:.6f} GPU days)'
        )
        print(f'initialize teacher pool time: {elapsed:.3f} s')

    def _teacher_kd_weights(self, scores_k):
        kd_mode = getattr(self, "kd_mode", "original")
        if kd_mode == "single_best":
            weights_k = torch.zeros_like(scores_k)
            best_idx = getattr(self, "best_teacher_index", None)
            if best_idx is None or best_idx < 0 or best_idx >= scores_k.numel():
                best_idx = 0
            weights_k[int(best_idx)] = 1.0
            return weights_k.detach()
        if kd_mode == "average":
            return (torch.ones_like(scores_k) / scores_k.numel()).detach()
        if kd_mode == "original":
            if float(scores_k.sum()) <= 0.0:
                weights_k = torch.ones_like(scores_k) / scores_k.numel()
            else:
                weights_k = scores_k / scores_k.sum()
            return weights_k.detach()
        raise ValueError(f"unknown kd_mode: {kd_mode}")

    def _select_best_teacher_index(self, arch_vector):
        if not getattr(self, "teacher_pool", None):
            return []
        if not getattr(self, "teacher_arch", None):
            return []

        arch_array = np.asarray(arch_vector, dtype=float)
        student_disc = update.vectorized_discretize(arch_array, self.min_, self.max_)
        student_cells = self._split_normal_cell_codes(student_disc)
        if not student_cells:
            return []

        num_cells = len(student_cells)
        best_teachers = [-1] * num_cells

        for pos in range(num_cells):
            s_code = np.asarray(student_cells[pos], dtype=float)
            if s_code.size == 0:
                continue
            best_idx = None
            best_dist = None
            for t_idx, t_arch in enumerate(self.teacher_arch):
                t_cells = self._split_normal_cell_codes(t_arch)
                if pos >= len(t_cells):
                    continue
                t_code = np.asarray(t_cells[pos], dtype=float)
                if t_code.size == 0:
                    continue
                L = min(s_code.size, t_code.size)
                if L == 0:
                    continue
                diff = s_code[:L] - t_code[:L]
                dist = float(np.sqrt(np.sum(diff * diff)))
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = t_idx
            if best_idx is not None:
                best_teachers[pos] = best_idx
        # best_teachers = [a1,a2,a3], 
        # a1:学生中第 1 个 normal cell 对应idx为a1的教师中第 1 个 normal cell 的索引
        # a2:学生中第 2 个 normal cell 对应idx为a2的教师中第 2 个 normal cell 的索引
        # a3:学生中第 3 个 normal cell 对应idx为a3的教师中第 3 个 normal cell 的索引
        return best_teachers   

    def _init_student_from_teacher(self, model, arch_vector):
        if not getattr(self, "teacher_pool", None):
            return
        if not hasattr(self, "teacher_cell_indices"):
            return

        best_teachers = self._select_best_teacher_index(arch_vector)
        if not best_teachers:
            return

        student_indices, _ = self._extract_block_features(model)
        if not student_indices:
            return

        for pos, s_cell_idx in enumerate(student_indices):
            if pos >= len(best_teachers):
                break
            t_idx = best_teachers[pos]
            if t_idx is None or t_idx < 0 or t_idx >= len(self.teacher_pool):
                continue
            t_indices = self.teacher_cell_indices[t_idx]
            if pos >= len(t_indices):
                continue
            t_cell_idx = t_indices[pos]
            teacher_model = self.teacher_pool[t_idx]
            # 将教师模型临时移到GPU进行参数复制
            teacher_model.to(self.device)
            self._init_cell_from_teacher_cell(model, teacher_model, s_cell_idx, t_cell_idx)
            # 复制完成后移回CPU节省内存
            teacher_model.cpu()

    def _evaluate_architecture_with_teachers(self, arch_vector, use_structural_transfer=True, use_kd_loss=True):
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        arch_array = np.asarray(arch_vector, dtype=float)
        discrete_arch = update.vectorized_discretize(arch_array, self.min_, self.max_)

        model = NetworkCifar(discrete_arch.tolist(), **self._network_kwargs())

        model.to(self.device)

        total_params = 0
        for p in model.parameters():
            total_params += p.numel()
        fit_1 = total_params / 1e6

        if use_structural_transfer and getattr(self, "teacher_pool", None):
            self._init_student_from_teacher(model, arch_vector)

        use_cached_teacher_stage = bool(use_kd_loss and self._has_teacher_stage_cache())
        if use_kd_loss and getattr(self, "teacher_pool", None) and not use_cached_teacher_stage:
            raise RuntimeError(
                "KD teacher stage cache is required after teacher_pool is initialized. "
                "Rebuild teacher_stage_cache instead of falling back to realtime teacher forward."
            )
        if use_cached_teacher_stage:
            train_loader = self._build_shared_train_loader(shuffle=True, return_index=True)
        else:
            base_loader = self.input_train
            ratio = getattr(self, "student_subset_ratio", 1.0)
            if ratio <= 0.0 or ratio >= 1.0:
                train_loader = base_loader
            else:
                dataset = base_loader.dataset
                total_size = len(dataset)
                subset_size = max(1, int(total_size * float(ratio)))
                indices = np.random.choice(total_size, subset_size, replace=False)
                subset = Subset(dataset, indices.tolist())
                batch_size = base_loader.batch_size
                num_workers = base_loader.num_workers
                pin_memory = getattr(base_loader, "pin_memory", True)
                train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory)

        valid_loader = self.input_valid
        epochs = 1
        train_criterion, eval_criterion, optimizer, scheduler = build_train_Optimizer_Loss(
            model, epochs=epochs, device=self.device
        )

        kd_loss_fn = nn.KLDivLoss(reduction="batchmean")
        T = self.T
        lambda_kd = self.lambda_kd

        last_acc = 0.0
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                inputs, targets, batch_positions = self._unpack_train_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                logits_s, s_stage_feats, aux_logits_s = self._forward_with_stage_features(model, inputs)
                ce_loss = train_criterion(logits_s, targets)
                if aux_logits_s is not None:
                    ce_loss = ce_loss + getattr(self, "search_aux_weight", 0.4) * train_criterion(aux_logits_s, targets)
                if use_cached_teacher_stage:
                    if batch_positions is None:
                        raise RuntimeError("KD teacher stage cache requires indexed training batches.")
                    t_stage_vecs_all = self._get_cached_teacher_stage_vectors(batch_positions)

                    kd_loss = ce_loss.new_tensor(0.0)
                    num_stages = min([len(s_stage_feats)] + [len(v) for v in t_stage_vecs_all])
                    s_feat = s_vec = t_vec_list_raw = s_vec_use = p_s_k = None
                    scores_k_list = p_t_list = scores_k = weights_k = None
                    log_p_s_k = stage_kd = None
                    for k in range(num_stages):
                        s_feat = s_stage_feats[k]
                        s_vec = F.adaptive_avg_pool2d(s_feat, 1).view(s_feat.size(0), -1)
                        t_vec_list_raw = [stage_vecs[k] for stage_vecs in t_stage_vecs_all]

                        c_s = s_vec.size(1)
                        c_t_list = [tv.size(1) for tv in t_vec_list_raw]
                        common_dim = min([c_s] + c_t_list)
                        if common_dim <= 0:
                            continue

                        s_vec_use = s_vec[:, :common_dim]
                        p_s_k = F.softmax(s_vec_use.detach() / T, dim=1)

                        scores_k_list = []
                        p_t_list = []
                        for tv in t_vec_list_raw:
                            tv_use = tv[:, :common_dim]
                            p_t = F.softmax(tv_use / T, dim=1)
                            p_t_list.append(p_t)
                            score_j = (p_s_k * p_t).sum(dim=1).sum()
                            scores_k_list.append(score_j)

                        scores_k = torch.stack(scores_k_list, dim=0)
                        weights_k = self._teacher_kd_weights(scores_k)

                        log_p_s_k = F.log_softmax(s_vec_use / T, dim=1)
                        stage_kd = ce_loss.new_tensor(0.0)
                        for j, p_t_j_k in enumerate(p_t_list):
                            kd_j_k = kd_loss_fn(log_p_s_k, p_t_j_k) * (T * T)
                            stage_kd = stage_kd + weights_k[j] * kd_j_k
                        kd_loss = kd_loss + stage_kd

                    loss = ce_loss + lambda_kd * kd_loss
                else:
                    loss = ce_loss
                loss.backward()
                optimizer.step()

                logits_s = s_stage_feats = aux_logits_s = ce_loss = loss = None
                if use_cached_teacher_stage:
                    kd_loss = t_stage_vecs_all = None
                    s_feat = s_vec = t_vec_list_raw = s_vec_use = p_s_k = None
                    scores_k_list = p_t_list = scores_k = weights_k = None
                    log_p_s_k = stage_kd = p_t_j_k = kd_j_k = None
                inputs = targets = None
            if scheduler is not None:
                scheduler.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    outputs = logits = preds = inputs = targets = None
            acc = correct / total if total > 0 else 0.0
            last_acc = acc

        fit_2 = -last_acc
        print(f'candidate objectives: f1(params_M)={fit_1:.6f}, f2(-valid_acc)={fit_2:.6f}')

        try:
            train_loader = valid_loader = None
            train_criterion = eval_criterion = optimizer = scheduler = None
            del model
            self._clear_runtime_cache()
        except Exception:
            pass

        return [fit_1, fit_2]

    def _evaluate_architecture_with_teachers_no_transfer(self, arch_vector):
        return self._evaluate_architecture_with_teachers(arch_vector, use_structural_transfer=False)

    def _evaluate_architecture_with_teachers_transfer_no_kd(self, arch_vector):
        return self._evaluate_architecture_with_teachers(
            arch_vector,
            use_structural_transfer=True,
            use_kd_loss=False,
        )
