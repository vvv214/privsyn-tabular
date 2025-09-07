###################################################
#                                                 #
#                GUM's main procedure             #
#                                                 #
###################################################

import logging

import numpy as np
import pandas as pd


class RecordUpdate:
    def __init__(self, domain, num_records, init_df=None):
        self.logger = logging.getLogger("record_update")
        
        self.domain = domain
        self.num_records = num_records
        self.df = init_df

    def initialize_records(self, iterate_keys, method="random", singletons=None):
        #ZL: Danyu found this bug
        #self.records = np.empty([self.num_records, len(self.domain.attrs)], dtype=np.uint32)
        if self.df is None:
            self.records = np.zeros([self.num_records, len(self.domain.attrs)], dtype=np.uint32)
            self.df = pd.DataFrame(self.records, columns=self.domain.attrs)

            for attr in self.domain.attrs:
                if method == "random":
                    self.df[attr] = np.random.randint(0, self.domain.config[attr], size=self.num_records)
                elif method == "singleton":
                    if attr in singletons:
                        self.df[attr] = self.generate_singleton_records(singletons[attr])
                    else:
                        self.df[attr] = np.random.randint(0, self.domain.config[attr], size=self.num_records)

        self.error_tracker = pd.DataFrame(index=iterate_keys)


    def generate_singleton_records(self, singleton):
        # Vectorized allocation according to cumulative distribution with exact total count
        dist_cumsum = np.cumsum(singleton.normalize_count)
        boundaries = np.rint(dist_cumsum * self.num_records).astype(int)
        counts = np.diff(np.concatenate(([0], boundaries)))
        # Ensure non-negative and exact size
        counts[counts < 0] = 0
        deficit = self.num_records - int(counts.sum())
        if deficit != 0:
            # Adjust the largest bin by the deficit to maintain total size
            arg = int(np.argmax(counts)) if counts.size > 0 else 0
            counts[arg] = max(0, counts[arg] + deficit)

        record = np.repeat(np.arange(counts.size, dtype=np.uint32), counts)
        np.random.shuffle(record)
        # In rare rounding edge-cases, pad or trim to exact length
        if record.size < self.num_records:
            pad = np.full(self.num_records - record.size, record[-1] if record.size > 0 else 0, dtype=np.uint32)
            record = np.concatenate([record, pad])
        elif record.size > self.num_records:
            record = record[: self.num_records]

        return record

    def update_records_main(self, marg, alpha):
        ##################################### deal with cells to be boosted #######################################
        # deal with the cell that synthesize_marginal != 0 and synthesize_marginal < actual_marginal
        self.cell_under_indices = \
            np.where((self.synthesize_marginal < self.actual_marginal) & (self.synthesize_marginal != 0))[0]
        
        ratio_add = np.minimum((self.actual_marginal[self.cell_under_indices] - self.synthesize_marginal[
            self.cell_under_indices]) / self.synthesize_marginal[self.cell_under_indices],
            np.full(self.cell_under_indices.shape[0], alpha))
        self.num_add = self.normal_rounding(ratio_add * self.synthesize_marginal[self.cell_under_indices] * self.num_records)
        
        # deal with the case synthesize_marginal == 0 and actual_marginal != 0
        self.cell_zero_indices = np.where((self.synthesize_marginal == 0) & (self.actual_marginal != 0))[0]
        self.num_add_zero = self.normal_rounding(alpha * self.actual_marginal[self.cell_zero_indices] * self.num_records)

        #################################### deal with cells to be reduced ########################################
        # determine the number of records to be removed
        self.cell_over_indices = np.where(self.synthesize_marginal > self.actual_marginal)[0]
        num_add_total = np.sum(self.num_add) + np.sum(self.num_add_zero)

        beta = self.find_optimal_beta(num_add_total, self.cell_over_indices)
        ratio_reduce = np.minimum(
            (self.synthesize_marginal[self.cell_over_indices] - self.actual_marginal[self.cell_over_indices])
            / self.synthesize_marginal[self.cell_over_indices], np.full(self.cell_over_indices.shape[0], beta))
        self.num_reduce = self.normal_rounding(
            ratio_reduce * self.synthesize_marginal[self.cell_over_indices] * self.num_records).astype(int)

        self.logger.debug("alpha: %s | beta: %s" % (alpha, beta))
        self.logger.debug("num_boost: %s | num_reduce: %s" % (num_add_total, np.sum(self.num_reduce)))

        # calculate some general variables
        self.encode_records = np.matmul(self.records[:, marg.attributes_index], marg.encode_num)
        self.encode_records_sort_index = np.argsort(self.encode_records)
        self.encode_records = self.encode_records[self.encode_records_sort_index]

    def determine_throw_indices(self):
        valid_indices = np.nonzero(self.num_reduce)[0]
        valid_cell_over_indices = self.cell_over_indices[valid_indices]
        valid_cell_num_reduce = self.num_reduce[valid_indices]
        left = np.searchsorted(self.encode_records, valid_cell_over_indices, side="left")
        right = np.searchsorted(self.encode_records, valid_cell_over_indices, side="right")

        total = int(np.sum(valid_cell_num_reduce))
        self.records_throw_indices = np.zeros(total, dtype=np.uint32)

        # Build selections per cell with permutation to avoid Python loops assigning piecewise
        selections = []
        for i in range(valid_cell_over_indices.size):
            start, end = left[i], right[i]
            match_indices = self.encode_records_sort_index[start:end]
            k = int(valid_cell_num_reduce[i])
            # Keep replace=False as original semantics; assume enough matches
            if k > 0:
                perm = np.random.permutation(match_indices)
                selections.append(perm[:k])
        if selections:
            self.records_throw_indices[:] = np.concatenate(selections)
        np.random.shuffle(self.records_throw_indices)
    
    def complete_partial_ratio(self, marg, complete_ratio):
        num_complete = np.rint(complete_ratio * self.num_add).astype(int)
        num_partial = np.rint((1 - complete_ratio) * self.num_add).astype(int)

        valid_mask = (num_complete + num_partial) > 0
        if not np.any(valid_mask):
            return

        valid_indices = np.nonzero(valid_mask)[0]
        num_complete_v = num_complete[valid_indices]
        num_partial_v = num_partial[valid_indices]
        num_total_v = num_complete_v + num_partial_v

        available = self.records_throw_indices.shape[0]
        if int(np.sum(num_total_v)) > available:
            # Fallback to original per-cell loop to preserve semantics when not enough slots
            valid_cell_under_indices = self.cell_under_indices[valid_indices]
            left = np.searchsorted(self.encode_records, valid_cell_under_indices, side="left")
            right = np.searchsorted(self.encode_records, valid_cell_under_indices, side="right")

            for i, cell_index in enumerate(valid_cell_under_indices):
                match_records_indices = self.encode_records_sort_index[left[i]: right[i]]
                np.random.shuffle(match_records_indices)

                need = num_total_v[i]
                if self.records_throw_indices.shape[0] >= need:
                    k_c = num_complete_v[i]
                    k_p = num_partial_v[i]
                    if k_c:
                        self.records[self.records_throw_indices[:k_c]] = self.records[match_records_indices[:k_c]]
                    if k_p:
                        self.records[np.ix_(self.records_throw_indices[k_c:k_c + k_p], marg.attributes_index)] = marg.tuple_key[cell_index]
                    self.records_throw_indices = self.records_throw_indices[need:]
                else:
                    self.records[self.records_throw_indices] = self.records[match_records_indices[: self.records_throw_indices.size]]
            return

        # Vectorized path when we have enough slots
        starts = np.cumsum(num_total_v) - num_total_v
        ends = starts + num_total_v

        valid_cell_under_indices = self.cell_under_indices[valid_indices]
        left = np.searchsorted(self.encode_records, valid_cell_under_indices, side="left")
        right = np.searchsorted(self.encode_records, valid_cell_under_indices, side="right")

        # For each cell, permute matches once and slice required count
        per_cell_selected = []
        sel_lengths = []
        for i in range(valid_indices.size):
            match_slice = self.encode_records_sort_index[left[i]: right[i]]
            need = int(num_total_v[i])
            if match_slice.size == 0 or need == 0:
                per_cell_selected.append(np.array([], dtype=np.uint32))
                sel_lengths.append(0)
                continue
            perm = np.random.permutation(match_slice)
            picked = perm[: min(need, match_slice.size)]
            per_cell_selected.append(picked)
            sel_lengths.append(picked.size)

        total_sel = int(np.sum(sel_lengths))
        if total_sel == 0:
            return
        selected_flat = np.concatenate([arr for arr in per_cell_selected if arr.size > 0])

        # Full replacements (complete) with clamping to available selected items
        full_src = []
        full_dst = []
        offset = 0
        for i in range(valid_indices.size):
            kc = int(num_complete_v[i])
            L = int(sel_lengths[i])
            if kc == 0 or L == 0:
                offset += L
                continue
            kc_eff = min(kc, L)
            s = int(starts[i])
            src_slice = selected_flat[offset: offset + L]
            full_src.append(src_slice[:kc_eff])
            full_dst.append(self.records_throw_indices[s: s + kc_eff])
            offset += L
        used_full = 0
        if full_src:
            fs = np.concatenate(full_src)
            fd = np.concatenate(full_dst)
            used_full = fd.size
            self.records[fd] = self.records[fs]

        # Partial updates (set marg attributes to tuple_key), also clamped to remaining per-cell selected items
        part_src_vals = []
        part_dst_idx = []
        offset = 0
        for i in range(valid_indices.size):
            L = int(sel_lengths[i])
            kp = int(num_partial_v[i])
            kc = int(num_complete_v[i])
            if L == 0 or kp == 0:
                offset += L
                continue
            kc_eff = min(kc, L)
            kp_eff = min(kp, L - kc_eff)
            if kp_eff <= 0:
                offset += L
                continue
            s = int(starts[i])
            dst = self.records_throw_indices[s + kc_eff: s + kc_eff + kp_eff]
            part_dst_idx.append(dst)
            tk = np.tile(marg.tuple_key[valid_cell_under_indices[i]], (kp_eff, 1))
            part_src_vals.append(tk)
            offset += L
        used_part = 0
        if part_dst_idx:
            dst_all = np.concatenate(part_dst_idx)
            vals_all = np.vstack(part_src_vals)
            used_part = dst_all.size
            self.records[np.ix_(dst_all, marg.attributes_index)] = vals_all

        # Consume only the number of throw indices that were actually used
        self.records_throw_indices = self.records_throw_indices[(used_full + used_part) :]
    
    def handle_zero_cells(self, marg):
        # overwrite / partial when synthesize_marginal == 0
        if self.cell_zero_indices.size == 0:
            return
        k = self.num_add_zero.astype(int)
        if np.sum(k) == 0:
            return
        available = self.records_throw_indices.shape[0]
        total_k = int(np.sum(k))
        if total_k > available:
            # Fallback to original per-cell behavior
            for index, cell_index in enumerate(self.cell_zero_indices):
                num_partial = int(self.num_add_zero[index])
                eff = min(num_partial, self.records_throw_indices.shape[0])
                if eff > 0:
                    self.records[np.ix_(self.records_throw_indices[: eff], marg.attributes_index)] = \
                        np.tile(marg.tuple_key[cell_index], (eff, 1))
                self.records_throw_indices = self.records_throw_indices[eff:]
            return

        # Vectorized assignment
        starts = np.cumsum(k) - k
        part_dst_idx = []
        part_src_vals = []
        for i, cell_index in enumerate(self.cell_zero_indices):
            ki = int(k[i])
            if ki == 0:
                continue
            s = int(starts[i])
            dst = self.records_throw_indices[s: s + ki]
            part_dst_idx.append(dst)
            part_src_vals.append(np.tile(marg.tuple_key[cell_index], (ki, 1)))
        if part_dst_idx:
            dst_all = np.concatenate(part_dst_idx)
            vals_all = np.vstack(part_src_vals)
            self.records[np.ix_(dst_all, marg.attributes_index)] = vals_all
        self.records_throw_indices = self.records_throw_indices[total_k:]
    
    def find_optimal_beta(self, num_add_total, cell_over_indices):
        actual_marginal_under = self.actual_marginal[cell_over_indices]
        synthesize_marginal_under = self.synthesize_marginal[cell_over_indices]
        
        lower_bound = 0.0
        upper_bound = 1.0
        beta = 0.0
        current_num = 0.0
        iteration = 0
        
        while abs(num_add_total - current_num) >= 1.0:
            beta = (upper_bound + lower_bound) / 2.0
            current_num = np.sum(
                np.minimum((synthesize_marginal_under - actual_marginal_under) / synthesize_marginal_under,
                np.full(cell_over_indices.shape[0], beta)) * synthesize_marginal_under * self.records.shape[0])
            
            if current_num < num_add_total:
                lower_bound = beta
            elif current_num > num_add_total:
                upper_bound = beta
            else:
                return beta

            iteration += 1
            if iteration > 50:
                self.logger.warning("cannot find the optimal beta")
                break
        
        return beta
    

    def update_records_before(self, marg, marg_key, iteration, mute=False):
        self.actual_marginal = marg.normalize_count
        self.synthesize_marginal = self.synthesize_marginal_distribution(marg)
        
        l1_error = self._l1_distance(self.actual_marginal, self.synthesize_marginal)

        #ZL: exception because marg_key should be a MultiIndex, as it's a tuple
        #self.error_tracker.loc[marg_key, "%s-before" % (iteration,)] = l1_error
        #vk = pd.MultiIndex.from_tuples([marg_key])
        #self.error_tracker.loc[vk, "%s-before" % (iteration,)] = l1_error
        #ZL: Danyu found changing marg_key (tuple) to a list solve
        
        self.error_tracker.loc[[marg_key], "%s-before" % (iteration,)] = l1_error
        
        if not mute:
            self.logger.info("the l1 error before updating is %s" % (l1_error,))

    def update_records_after(self, marg, marg_key, iteration, mute=False):
        self.synthesize_marginal = self.synthesize_marginal_distribution(marg)
        
        l1_error = self._l1_distance(self.actual_marginal, self.synthesize_marginal)

        #ZL: exception because marg_key should be a MultiIndex, as it's a tuple
        #self.error_tracker.loc[marg_key, "%s-before" % (iteration,)] = l1_error
        #vk = pd.MultiIndex.from_tuples([marg_key])
        #self.error_tracker.loc[vk, "%s-before" % (iteration,)] = l1_error
        #ZL: Danyu found changing marg_key (tuple) to a list solve

        self.error_tracker.loc[[marg_key], "%s-after" % (iteration,)] = l1_error
        
        if not mute:
            self.logger.info("the l1 error after updating is %s" % (l1_error,))
        
        # if iteration == 0:
        #     self.logger.info("shuffling records")
        #     np.random.shuffle(self.records)
    
    def synthesize_marginal_distribution(self, marg):
        count = marg.count_records_general(self.records)
        # count_matrix = marg.calculate_count_matrix_general(count)
        
        return marg.calculate_normalize_count_general(count)

    def stochastic_rounding(self, vector):
        ret_vector = np.zeros(vector.size)
        rand = np.random.rand(vector.size)

        integer = np.floor(vector)
        decimal = vector - integer

        ret_vector[rand > decimal] = np.floor(decimal[rand > decimal])
        ret_vector[rand < decimal] = np.ceil(decimal[rand < decimal])
        ret_vector += integer

        return ret_vector
    
    def normal_rounding(self, vector):
        return np.round(vector)

    def _l1_distance(self, t1, t2):
        assert len(t1) == len(t2)
    
        return np.sum(np.absolute(t1 - t2))
