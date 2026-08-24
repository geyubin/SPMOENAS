import numpy as np
import random
import gc
import time
from fitness import *
import init
import update

from MopsoExt import MopsoExtMixin


class Mopso(MopsoExtMixin):
    def __init__(self, particals, w, c1, c2, max_, min_, thresh, input_train, input_valid,
                 mesh_div=10, device=None, teacher_num=5, teacher_epochs=50,
                 subset_ratio=0.5, num_classes=10,
                 use_structural_transfer=True, use_kd=True,
                 teacher_subset_ratio=None, student_subset_ratio=None,
                 lambda_kd=1.0, T=1.0, use_proposed_mopso=True,
                 model_channels=64, model_stem_type="cifar",
                 search_use_aux=False, search_aux_weight=0.4,
                 search_aux_head_type="cifar", kd_mode="original"):
        self.w, self.c1, self.c2 = w, c1, c2
        self.mesh_div = mesh_div
        self.particals = particals
        self.thresh = thresh
        self.max_ = max_
        self.min_ = min_
        self.max_v = (max_ - min_) * 0.05
        self.min_v = (max_ - min_) * 0.05 * (-1)
        self.input_train = input_train
        self.input_valid = input_valid
        self.device = device
        self.num_classes = num_classes
        self.model_channels = int(model_channels)
        self.model_stem_type = model_stem_type
        self.search_use_aux = bool(search_use_aux)
        self.search_aux_weight = float(search_aux_weight)
        self.search_aux_head_type = search_aux_head_type
        self.teacher_num = teacher_num
        self.teacher_epochs = max(0, int(teacher_epochs))
        if teacher_subset_ratio is None:
            teacher_subset_ratio = subset_ratio
        if student_subset_ratio is None:
            student_subset_ratio = subset_ratio
        self.teacher_subset_ratio = float(teacher_subset_ratio)
        self.student_subset_ratio = float(student_subset_ratio)
        self.lambda_kd = float(lambda_kd)
        self.T = float(T)
        self.kd_mode = kd_mode
        self.use_structural_transfer = use_structural_transfer
        self.use_kd = use_kd
        self.use_proposed_mopso = use_proposed_mopso
        self.teacher_pool = []
        self.teacher_arch = []
        self.teacher_pretrain_elapsed_time = 0.0
        self.teacher_pool_elapsed_time = 0.0
        self.teacher_cell_outputs = []
        self.teacher_cell_weights = []
        self.teacher_cell_heads = []
        self.teacher_block_feats = []
        self.teacher_cell_indices = []
        self.teacher_stage_cache = []
        self.teacher_valid_accs = []
        self.best_teacher_index = None
        self._shared_train_indices = None
        self._shared_train_tensor_cache = None
        self.dom_age = np.zeros(self.particals, dtype=int)
        self.dom_potential = np.zeros(self.particals, dtype=float)
        self.low_potential_mask = np.zeros(self.particals, dtype=bool)
        self.pref_particles = np.zeros((0, max_.shape[0]))
        self.current_iter = 0
        self.max_iter = 1

    @staticmethod
    def _filter_nondominated_points(points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0:
            return points

        keep = np.ones(points.shape[0], dtype=bool)
        for i in range(points.shape[0]):
            if not keep[i]:
                continue
            for j in range(points.shape[0]):
                if i == j:
                    continue
                dominates_i = np.all(points[j] <= points[i]) and np.any(points[j] < points[i])
                if dominates_i:
                    keep[i] = False
                    break
        return points[keep]

    @classmethod
    def _hypervolume_minimization(cls, points, ref_point):
        points = np.asarray(points, dtype=float)
        ref_point = np.asarray(ref_point, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0:
            return 0.0

        points = points[np.all(points < ref_point, axis=1)]
        if points.shape[0] == 0:
            return 0.0

        points = cls._filter_nondominated_points(points)
        dim = points.shape[1]
        if dim == 1:
            return max(0.0, float(ref_point[0] - np.min(points[:, 0])))

        coords = np.unique(points[:, 0])
        coords = coords[coords < ref_point[0]]
        if coords.size == 0:
            return 0.0

        hv = 0.0
        for idx, left in enumerate(coords):
            right = coords[idx + 1] if idx + 1 < coords.size else ref_point[0]
            width = right - left
            if width <= 0:
                continue
            active = points[points[:, 0] <= left]
            hv += width * cls._hypervolume_minimization(active[:, 1:], ref_point[1:])
        return float(hv)

    def _compute_archive_hv(self):
        archive_fit = np.asarray(getattr(self, "archive_fitness", []), dtype=float)
        if archive_fit.ndim != 2 or archive_fit.shape[0] == 0:
            return 0.0, 0

        min_vals = archive_fit.min(axis=0)
        max_vals = archive_fit.max(axis=0)
        spans = max_vals - min_vals
        normalized = np.zeros_like(archive_fit, dtype=float)
        valid_span = spans > 1e-12
        normalized[:, valid_span] = (archive_fit[:, valid_span] - min_vals[valid_span]) / spans[valid_span]
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        ref_point = np.full(normalized.shape[1], 1.1, dtype=float)
        hv_value = self._hypervolume_minimization(normalized, ref_point)
        return hv_value, archive_fit.shape[0]

    def evaluation_fitness(self, in_):
        fitness_curr = []
        for i in range(len(in_)):
            if self.use_kd:
                if self.use_structural_transfer:
                    fitness_curr.append(self._evaluate_architecture_with_teachers(in_[i]))
                else:
                    '''消融-2：不用结构感知权重'''
                    fitness_curr.append(self._evaluate_architecture_with_teachers_no_transfer(in_[i]))
            else:
                if self.use_structural_transfer:
                    '''消融3：不用多阶段集成多教师蒸馏训练'''
                    fitness_curr.append(self._evaluate_architecture_with_teachers_transfer_no_kd(in_[i]))
                else:
                    '''仅采用PPMOPSO'''
                    fitness_curr.append(evaluate_architecture_without_teachers(in_[i], self.input_train,
                                                                               self.input_valid, self.min_, self.max_,
                                                                               self.device, self.num_classes,
                                                                               self.model_channels,
                                                                               self.model_stem_type,
                                                                               self.search_use_aux,
                                                                               self.search_aux_weight,
                                                                               self.search_aux_head_type))
        self.fitness_ = np.array(fitness_curr)

    def initialize(self):

        self.in_ = init.init_designparams(self.particals, self.min_, self.max_)

        self.v_ = init.init_v(self.particals, self.min_v, self.max_v)

        self.evaluation_fitness(self.in_)

        if self.use_proposed_mopso:
            dominated = update.compute_dominated_mask(self.fitness_)
            self.dom_age[:] = 0
            self.dom_age[dominated] = 1
            potential_dom = update.compute_potential_for_dominated(self.fitness_[dominated], self.dom_age[dominated])
            self.dom_potential[:] = 0.0
            if potential_dom.size > 0:
                dom_indices = np.where(dominated)[0]
                self.dom_potential[dom_indices] = potential_dom
                order = np.argsort(-potential_dom)
                split = potential_dom.size // 2
                low_dom_indices = dom_indices[order[:split]]
                self.low_potential_mask[:] = False
                self.low_potential_mask[low_dom_indices] = True
        else:
            self.dom_age[:] = 0
            self.dom_potential[:] = 0.0
            self.low_potential_mask[:] = False

        self.in_p, self.fitness_p = init.init_pbest(self.in_, self.fitness_)

        self.archive_in, self.archive_fitness = init.init_archive(self.in_, self.fitness_)

        self.in_g, self.fitness_g = init.init_gbest(self.archive_in, self.archive_fitness, self.mesh_div, self.min_,
                                                    self.max_, self.particals)

        if self.use_proposed_mopso:
            self.pref_particles = update.compute_sparse_preference_particles(self.archive_in, self.archive_fitness, self.mesh_div)
        else:
            self.pref_particles = np.zeros((0, self.max_.shape[0]))

        if self.use_structural_transfer or self.use_kd:
            self._initialize_teacher_pool()

    def update_(self):
        dominated_prev = self.dom_age > 0
        if self.use_proposed_mopso:
            low_dom_prev = self.low_potential_mask & dominated_prev
        else:
            low_dom_prev = np.zeros(self.particals, dtype=bool)
        high_dom_prev = dominated_prev & (~low_dom_prev)
        non_dom_prev = ~dominated_prev

        gbest_for_update = np.array(self.in_g, copy=True)

        if self.archive_in is not None and self.archive_in.size > 0:
            gbest_neighbors = update.compute_neighbor_gbest_from_archive(
                self.archive_in,
                self.archive_fitness,
                self.mesh_div,
                self.particals,
            )
            if non_dom_prev.any():
                gbest_for_update[non_dom_prev] = gbest_neighbors[non_dom_prev]

        if self.use_proposed_mopso and self.pref_particles is not None and self.pref_particles.size > 0:
            low_indices = np.where(low_dom_prev)[0]
            if low_indices.size > 0:
                num_pref = self.pref_particles.shape[0]
                rand_idx = np.random.randint(0, num_pref, size=low_indices.size)
                gbest_for_update[low_indices] = self.pref_particles[rand_idx]

        v_prev = np.array(self.v_, copy=True)
        v_new = update.update_v(self.v_, self.min_v, self.max_v, self.in_, self.in_p, gbest_for_update, self.w, self.c1, self.c2)

        if self.use_proposed_mopso:
            low_indices = np.where(low_dom_prev)[0]
            if low_indices.size > 0 and self.max_iter > 0:
                t = self.current_iter + 1
                t_max = self.max_iter
                beta = float(t) / float(t_max)
                for idx in low_indices:
                    r1 = random.random()
                    inertia = self.w * v_prev[idx]
                    social = self.c1 * r1 * (gbest_for_update[idx] - self.in_[idx])
                    cognitive = beta * (self.in_p[idx] - self.in_[idx])
                    v_i = inertia + social + cognitive
                    for j in range(v_i.shape[0]):
                        if v_i[j] < self.min_v[j]:
                            v_i[j] = self.min_v[j]
                        if v_i[j] > self.max_v[j]:
                            v_i[j] = self.max_v[j]
                    v_new[idx] = v_i

        self.v_ = v_new

        self.in_ = update.update_in(self.in_, self.v_, self.min_, self.max_)
        self.evaluation_fitness(self.in_)

        dominated = update.compute_dominated_mask(self.fitness_)
        self.dom_age[~dominated] = 0
        self.dom_age[dominated] = self.dom_age[dominated] + 1
        if self.use_proposed_mopso:
            potential_dom = update.compute_potential_for_dominated(self.fitness_[dominated], self.dom_age[dominated])
            self.dom_potential[:] = 0.0
            if potential_dom.size > 0:
                dom_indices = np.where(dominated)[0]
                self.dom_potential[dom_indices] = potential_dom
                order = np.argsort(-potential_dom)
                split = potential_dom.size // 2
                low_dom_indices = dom_indices[order[:split]]
                self.low_potential_mask[:] = False
                self.low_potential_mask[low_dom_indices] = True
            else:
                self.low_potential_mask[:] = False

        self.in_p, self.fitness_p = update.update_pbest(self.in_, self.fitness_, self.in_p, self.fitness_p)
        self.archive_in, self.archive_fitness = update.update_archive(self.in_, self.fitness_,
                                                                                      self.archive_in,
                                                                                      self.archive_fitness,
                                                                                      self.thresh, self.mesh_div,
                                                                                      self.min_, self.max_, self.particals)
        self.in_g, self.fitness_g = update.update_gbest(self.archive_in, self.archive_fitness, self.mesh_div, self.min_,
                                                        self.max_, self.particals)

        if self.use_proposed_mopso:
            self.pref_particles = update.compute_sparse_preference_particles(self.archive_in, self.archive_fitness, self.mesh_div)
                                                                                
    def _evaluate_one_architecture(self, arch_vector):
        if self.use_kd:
            if self.use_structural_transfer:
                return self._evaluate_architecture_with_teachers(arch_vector)
            return self._evaluate_architecture_with_teachers_no_transfer(arch_vector)

        if self.use_structural_transfer:
            return self._evaluate_architecture_with_teachers_transfer_no_kd(arch_vector)

        return evaluate_architecture_without_teachers(
            arch_vector,
            self.input_train,
            self.input_valid,
            self.min_,
            self.max_,
            self.device,
            self.num_classes,
            self.model_channels,
            self.model_stem_type,
            self.search_use_aux,
            self.search_aux_weight,
            self.search_aux_head_type,
        )

    def evaluate_pareto_with_aux(self, pareto_archs, aux_weight=None, aux_head_type=None):
        archs = np.asarray(pareto_archs, dtype=float)
        if archs.size == 0:
            return np.zeros((0, 2), dtype=float)
        if archs.ndim == 1:
            archs = archs.reshape(1, -1)

        old_search_use_aux = self.search_use_aux
        old_search_aux_weight = self.search_aux_weight
        old_search_aux_head_type = self.search_aux_head_type

        self.search_use_aux = True
        if aux_weight is not None:
            self.search_aux_weight = float(aux_weight)
        if aux_head_type is not None:
            self.search_aux_head_type = aux_head_type

        print("start final Pareto auxiliary-head evaluation")
        aux_fitness = []
        try:
            for i, arch_vector in enumerate(archs):
                print(f"[Pareto aux eval] architecture index {i}")
                fit = self._evaluate_one_architecture(arch_vector)
                aux_fitness.append(fit)
                print(f"[Pareto aux eval] index {i} params_M {fit[0]} valid_acc {-fit[1]}")
        finally:
            self.search_use_aux = old_search_use_aux
            self.search_aux_weight = old_search_aux_weight
            self.search_aux_head_type = old_search_aux_head_type

        return np.asarray(aux_fitness, dtype=float)

    def done(self, cycle_):
        self.initialize()

        for i in range(cycle_):
            print('the current iteration is:', i)
            iter_start_time = time.time()
            
            self.current_iter = i
            self.max_iter = cycle_
            self.update_()
            archive_hv, archive_nd_count = self._compute_archive_hv()
            iter_elapsed_time = time.time() - iter_start_time
            print(
                f"[Round {i}] elapsed_time: {iter_elapsed_time:.3f} s "
                f"({iter_elapsed_time / 86400:.6f} GPU days), "
                f"Archive HV: {archive_hv:.6f}, archive non-dominated count: {archive_nd_count}"
            )

            # 定期清理垃圾回收
            if i % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return self.archive_in, self.archive_fitness
