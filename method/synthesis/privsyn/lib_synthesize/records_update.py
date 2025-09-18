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

        # lazy initialize error tracker without preset index to avoid index-type mismatches
        self.error_tracker = pd.DataFrame()


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
        valid_data_over_index_left = np.searchsorted(self.encode_records, valid_cell_over_indices, side="left")
        valid_data_over_index_right = np.searchsorted(self.encode_records, valid_cell_over_indices, side="right")
    
        valid_num_reduce = np.sum(valid_cell_num_reduce)
        self.records_throw_indices = np.zeros(valid_num_reduce, dtype=np.uint32)
        throw_pointer = 0
        
        for i, cell_index in enumerate(valid_cell_over_indices):
            match_records_indices = self.encode_records_sort_index[
                                    valid_data_over_index_left[i]: valid_data_over_index_right[i]]
            throw_indices = np.random.choice(match_records_indices, valid_cell_num_reduce[i], replace=False)
            
            self.records_throw_indices[throw_pointer:throw_pointer + valid_cell_num_reduce[i]] = throw_indices
            throw_pointer += valid_cell_num_reduce[i]

    def normal_rounding(self, arr):
        arr = np.rint(arr).astype(int)
        arr[arr < 0] = 0
        return arr

    def find_optimal_beta(self, num_add_total, cell_over_indices):
        # Simple heuristic: full reduction allowed
        if cell_over_indices.size == 0:
            return 0.0
        return 1.0

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

    def update_records_before(self, marg, marg_key, iteration, mute=False):
        marg.calculate_tuple_key()
        actual = marg.calculate_normalize_count()
        self.actual_marginal = actual
        synth_count = marg.count_records_general(self.records)
        synth = marg.calculate_normalize_count_general(synth_count)
        self.synthesize_marginal = synth
        err = float(np.abs(actual - synth).sum())
        col = f"{iteration}-before"
        key_tuple = tuple(marg_key) if not isinstance(marg_key, tuple) else marg_key
        key_label = "::".join(map(str, key_tuple))
        if key_label not in self.error_tracker.index:
            self.error_tracker.loc[key_label, :] = np.nan
        self.error_tracker.loc[key_label, col] = err

    def complete_partial_ratio(self, marg, ratio):
        need_map = {}
        for idx, c in zip(self.cell_under_indices, self.num_add):
            if c > 0:
                need_map[idx] = need_map.get(idx, 0) + int(c)
        for idx, c in zip(self.cell_zero_indices, self.num_add_zero):
            if c > 0:
                need_map[idx] = need_map.get(idx, 0) + int(c)

        if self.records_throw_indices.size == 0 or not need_map:
            return

        ptr = 0
        total_available = self.records_throw_indices.shape[0]
        for cell_index, need in need_map.items():
            if need <= 0:
                continue
            take = min(need, max(0, total_available - ptr))
            if take <= 0:
                break
            dst_idx = self.records_throw_indices[ptr: ptr + take]
            ptr += take
            self.records[np.ix_(dst_idx, marg.attributes_index)] = marg.tuple_key[cell_index]

    def update_records_after(self, marg, marg_key, iteration, mute=False):
        synth_count = marg.count_records_general(self.records)
        synth = marg.calculate_normalize_count_general(synth_count)
        actual = self.actual_marginal
        err = float(np.abs(actual - synth).sum())
        col = f"{iteration}-after"
        key_tuple = tuple(marg_key) if not isinstance(marg_key, tuple) else marg_key
        key_label = "::".join(map(str, key_tuple))
        if key_label not in self.error_tracker.index:
            self.error_tracker.loc[key_label, :] = np.nan
        self.error_tracker.loc[key_label, col] = err
