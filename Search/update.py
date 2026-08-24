import numpy as np
import random
import os
import sys
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pareto
import archive
from SearchSpace import *

def vectorized_discretize(in_array, min_vals, max_vals):
    in_array = np.asarray(in_array)
    min_vals = np.asarray(min_vals)
    max_vals = np.asarray(max_vals)

    n_classes = np.round(max_vals - min_vals + 1).astype(int)
    intervals = (max_vals - min_vals) / n_classes

    idx = ((in_array - min_vals) / intervals).astype(int)
    idx = np.clip(idx, 0, n_classes - 1)

    return (min_vals + idx).astype(int)

# 计算每个种群的支配粒子
def compute_dominated_mask(fitness_):
    fitness_ = np.asarray(fitness_)
    num = fitness_.shape[0]
    dominated = np.zeros(num, dtype=bool)
    for i in range(num):
        if dominated[i]:
            continue
        for j in range(num):
            if i == j:
                continue
            if not pareto.compare_(fitness_[i], fitness_[j]):
                dominated[i] = True
                break
    return dominated


def compute_potential_for_dominated(fitness_dom, ages_dom):
    fitness_dom = np.asarray(fitness_dom)
    ages_dom = np.asarray(ages_dom, dtype=float)
    if fitness_dom.size == 0:
        return np.array([], dtype=float)

    num_particles, num_obj = fitness_dom.shape
    ranks = np.zeros_like(fitness_dom, dtype=float)
    for j in range(num_obj):
        order = np.argsort(fitness_dom[:, j])   # np.argsort 按值从小到大返回元素的索引（最小化问题是从小到大）
        rank = np.empty(num_particles, dtype=float)  # 返回一个新的数组，元素为 float 类型且大小随机，大小为 num_particles
        rank[order] = np.arange(1, num_particles + 1, dtype=float)  # 在rank中，根据order的索引顺序赋值，从1开始递增（1为名次第一，目标值最小）
        ranks[:, j] = rank

    ages_dom = np.maximum(ages_dom, 1.0)
    c = (np.log(ages_dom) + np.sum(np.log(ranks), axis=1)) / (num_obj + 1.0)
    return c


def compute_sparse_preference_particles(archive_in, archive_fit, mesh_div):
    archive_in = np.asarray(archive_in)
    archive_fit = np.asarray(archive_fit)
    if archive_in.size == 0 or archive_fit.size == 0:
        return np.zeros((0, archive_in.shape[1] if archive_in.ndim == 2 else 0))

    num_points, num_obj = archive_fit.shape
    mins = archive_fit.min(axis=0)
    maxs = archive_fit.max(axis=0)
    span = maxs - mins
    span[span == 0] = 1.0  # 将span为0的维度设为1，避免除0，
    coords = ((archive_fit - mins) / span * mesh_div).astype(int)
    coords = np.clip(coords, 0, mesh_div - 1)  # 将coords限制在[0, mesh_div-1]范围内

    grid_map = {}
    for idx, coord in enumerate(coords):
        key = tuple(coord.tolist())  # 将coord转换为元组，作为字典的键
        grid_map.setdefault(key, []).append(idx)  # value 为落在该网格的非支配粒子索引列表

    if not grid_map:
        return np.zeros((0, archive_in.shape[1]))

    counts = {k: len(v) for k, v in grid_map.items()}   # len(v)为第k个网格的GCD值
    positive_counts = [c for c in counts.values() if c > 0]   # 去除GCD=0的网格
    if not positive_counts:
        return np.zeros((0, archive_in.shape[1]))
    min_count = min(positive_counts)
    sparse_keys = [k for k, c in counts.items() if c == min_count and c > 0]  # 获得稀疏网格的键（坐标）列表

    pref_list = []
    for key in sparse_keys:
        center_indices = grid_map[key]
        neighbor_keys = []
        # 遍历center_indices的3^M 个邻居偏移（上下左右、斜对角等），找到相邻网格的键（坐标）列表
        for offsets in itertools.product([-1, 0, 1], repeat=num_obj):
            if all(o == 0 for o in offsets):
                continue
            neighbor = []
            # key是一个元组，根据目标数遍历每一个相邻网格
            for d in range(num_obj):
                v = key[d] + offsets[d]
                if v < 0 or v >= mesh_div:
                    break
                neighbor.append(v)
            if len(neighbor) != num_obj:
                continue
            neighbor = tuple(neighbor)
            if neighbor in grid_map and counts[neighbor] > 0:
                neighbor_keys.append(neighbor)

        if not neighbor_keys:
            continue

        best_pair = None
        best_dist = -1.0
        # 寻找“中间无第三粒子”的最大距离粒子对，即稀疏区域上的偏好粒子
        for nkey in neighbor_keys:
            n_indices = grid_map[nkey]
            # 遍历center_indices中的每个粒子，与n_indices中的每个粒子计算距离
            for gi in center_indices:
                fg = archive_fit[gi]
                for ni in n_indices:
                    if ni == gi:
                        continue
                    fn = archive_fit[ni]
                    diff = fg - fn
                    dist = float(np.sqrt(np.sum(diff * diff)))   # 粒子对的欧式距离

                    lo = np.minimum(fg, fn)
                    hi = np.maximum(fg, fn)
                    between = ((archive_fit >= lo) & (archive_fit <= hi)).all(axis=1)
                    between[gi] = False
                    between[ni] = False
                    if np.any(between):
                        continue

                    if dist > best_dist:
                        best_dist = dist
                        best_pair = (gi, ni)

        if best_pair is not None:
            gi, ni = best_pair
            pref = 0.5 * (archive_in[gi] + archive_in[ni])
            pref_list.append(pref)

    if not pref_list:
        return np.zeros((0, archive_in.shape[1]))
    return np.vstack(pref_list)


def compute_neighbor_gbest_from_archive(archive_in, archive_fit, mesh_div, particals):
    archive_in = np.asarray(archive_in)
    archive_fit = np.asarray(archive_fit)
    if archive_in.size == 0 or archive_fit.size == 0:
        if archive_in.ndim == 2 and archive_in.shape[1] > 0:
            return np.tile(archive_in[0:1], (particals, 1))
        return np.zeros((particals, 0))

    num_points, num_obj = archive_fit.shape
    mins = archive_fit.min(axis=0)
    maxs = archive_fit.max(axis=0)
    span = maxs - mins
    span[span == 0] = 1.0
    coords = ((archive_fit - mins) / span * mesh_div).astype(int)
    coords = np.clip(coords, 0, mesh_div - 1)

    grid_map = {}
    for idx, coord in enumerate(coords):
        key = tuple(coord.tolist())
        grid_map.setdefault(key, []).append(idx)

    if not grid_map:
        indices = np.random.randint(0, archive_in.shape[0], size=particals)
        return archive_in[indices]

    dom_mask = compute_dominated_mask(archive_fit)
    counts = {}
    for k, v in grid_map.items():
        cnt = 0
        for idx in v:
            if not dom_mask[idx]:
                cnt += 1
        counts[k] = cnt

    neighbor_keys_map = {}
    for key in grid_map.keys():
        neighbor_keys = []
        for offsets in itertools.product([-1, 0, 1], repeat=num_obj):
            if all(o == 0 for o in offsets):
                continue
            neighbor = []
            for d in range(num_obj):
                v = key[d] + offsets[d]
                if v < 0 or v >= mesh_div:
                    break
                neighbor.append(v)
            if len(neighbor) != num_obj:
                continue
            neighbor = tuple(neighbor)
            if neighbor in grid_map and counts.get(neighbor, 0) > 0:
                neighbor_keys.append(neighbor)
        neighbor_keys_map[key] = neighbor_keys

    leaders_per_archive = np.zeros_like(archive_in, dtype=float)
    for i in range(num_points):
        key = tuple(coords[i].tolist())
        center_indices = grid_map.get(key, [i])
        center_non_dom = [idx for idx in center_indices if not dom_mask[idx]]
        neighbor_keys = neighbor_keys_map.get(key, [])
        if neighbor_keys:
            neighbor_counts = [counts.get(nk, 0) for nk in neighbor_keys]
            positive_counts = [c for c in neighbor_counts if c > 0]
            if positive_counts:
                min_count = min(positive_counts)
                sparse_keys = [nk for nk in neighbor_keys if counts.get(nk, 0) == min_count and counts.get(nk, 0) > 0]
                if sparse_keys:
                    chosen_key = random.choice(sparse_keys)
                    candidates = [idx for idx in grid_map[chosen_key] if not dom_mask[idx]]
                    if candidates:
                        sel_idx = random.choice(candidates)
                    elif center_non_dom:
                        sel_idx = random.choice(center_non_dom)
                    else:
                        sel_idx = random.choice(center_indices)
                else:
                    if center_non_dom:
                        sel_idx = random.choice(center_non_dom)
                    else:
                        sel_idx = random.choice(center_indices)
            else:
                if center_non_dom:
                    sel_idx = random.choice(center_non_dom)
                else:
                    sel_idx = random.choice(center_indices)
        else:
            if center_non_dom:
                sel_idx = random.choice(center_non_dom)
            else:
                sel_idx = random.choice(center_indices)
        leaders_per_archive[i] = archive_in[sel_idx]

    dim = archive_in.shape[1]
    gbest = np.zeros((particals, dim), dtype=float)
    if num_points >= particals:
        indices = np.random.randint(0, num_points, size=particals)
        gbest[:] = leaders_per_archive[indices]
    else:
        for i in range(particals):
            gbest[i] = leaders_per_archive[i % num_points]
    return gbest


def update_v(v_, v_min, v_max, in_, in_pbest, in_gbest, w, c1, c2):
    v_temp = w * v_ + c1 * random.random() * (in_pbest - in_) + c2 * random.random() * (in_gbest - in_)

    for i in range(v_temp.shape[0]):
        for j in range(v_temp.shape[1]):
            if v_temp[i, j] < v_min[j]:
                v_temp[i, j] = v_min[j]
            if v_temp[i, j] > v_max[j]:
                v_temp[i, j] = v_max[j]
    return v_temp


def update_in(in_, v_, in_min, in_max):
    in_temp = in_ + v_

    for i in range(in_temp.shape[0]):
        for j in range(in_temp.shape[1]):
            if in_temp[i, j] < in_min[j]:
                in_temp[i, j] = in_min[j]
            if in_temp[i, j] > in_max[j]:
                in_temp[i, j] = in_max[j]
    return in_temp


def compare_pbest(in_indiv, pbest_indiv):
    num_greater = 0
    num_less = 0
    for i in range(len(in_indiv)):
        if in_indiv[i] < pbest_indiv[i]:
            num_greater = num_greater + 1
        if in_indiv[i] > pbest_indiv[i]:
            num_less = num_less + 1

    if num_greater > 0 and num_less == 0:
        return True

    elif num_greater == 0 and num_less > 0:
        return False
    else:

        random_ = random.uniform(0.0, 1.0)
        if random_ > 0.5:
            return True
        else:
            return False


def update_pbest(in_, fitness_, in_pbest, out_pbest):
    for i in range(out_pbest.shape[0]):

        if compare_pbest(fitness_[i], out_pbest[i]):
            out_pbest[i] = fitness_[i]
            in_pbest[i] = in_[i]
    return in_pbest, out_pbest


def update_archive(in_, fitness_, archive_in, archive_fitness, thresh, mesh_div, min_, max_, particals,):
    pareto_1 = pareto.Pareto_(in_, fitness_)
    curr_in, curr_fit = pareto_1.pareto()
    in_new = np.concatenate((archive_in, curr_in), axis=0)
    fitness_new = np.concatenate((archive_fitness, curr_fit), axis=0)
    pareto_2 = pareto.Pareto_(in_new, fitness_new)
    curr_archiving_in, curr_archiving_fit = pareto_2.pareto()
    if (curr_archiving_in).shape[0] > thresh:
        clear_ = archive.clear_archiving(curr_archiving_in, curr_archiving_fit, mesh_div, min_, max_, particals)
        curr_archiving_in, curr_archiving_fit = clear_.clear_(thresh)
    return curr_archiving_in, curr_archiving_fit


def update_gbest(archiving_in, archiving_fit, mesh_div, min_, max_, particals):
    get_g = archive.get_gbest(archiving_in, archiving_fit, mesh_div, min_, max_, particals)
    return get_g.get_gbest()
