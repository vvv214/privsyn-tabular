import numpy as np

class Marginal:
    def __init__(self, marg_domain, dataset_domain):

        self.indicator = np.zeros(len(dataset_domain.attrs), dtype=np.uint8)
        self.indicator[[dataset_domain.attr_index_mapping[attr]
                        for attr in marg_domain.attrs]] = 1
        self.num_categories = np.array(dataset_domain.shape)

        self.attributes_index = np.nonzero(self.indicator)[0]
        self.attr_set = set(marg_domain.attrs)

        self.num_key = np.prod(self.num_categories[self.attributes_index])
        self.num_attributes = self.indicator.shape[0]
        self.ways = np.count_nonzero(self.indicator)

        self.encode_num = np.zeros(self.ways, dtype=np.uint32)
        self.cum_mul = np.zeros(self.ways, dtype=np.uint32)

        self.count = np.zeros(self.num_key)
        self.rho = 0.0

        self.calculate_encode_num(self.num_categories)

    def calculate_encode_num(self, num_categories):
        if self.ways != 0:
            categories_index = self.attributes_index

            categories_num = num_categories[categories_index]
            categories_num = np.roll(categories_num, 1)
            categories_num[0] = 1
            self.cum_mul = np.cumprod(categories_num)

            categories_num = num_categories[categories_index]
            categories_num = np.roll(categories_num, self.ways - 1)
            categories_num[-1] = 1
            categories_num = np.flip(categories_num)
            self.encode_num = np.flip(np.cumprod(categories_num))

    def calculate_tuple_key(self):
        self.tuple_key = np.zeros([self.num_key, self.ways], dtype=np.uint32)

        if self.ways != 0:
            for i in range(self.attributes_index.shape[0]):
                index = self.attributes_index[i]
                categories = np.arange(self.num_categories[index])
                column_key = np.tile(
                    np.repeat(categories, self.encode_num[i]), self.cum_mul[i])

                self.tuple_key[:, i] = column_key
        else:
            self.tuple_key = np.array([0], dtype=np.uint32)
            self.num_key = 1

    def count_records(self, records):
        encode_records = np.matmul(
            records[:, self.attributes_index], self.encode_num)
        encode_key, count_num = np.unique(encode_records, return_counts=True)
        indices = np.where(np.isin(np.arange(self.num_key), encode_key))[0]
        self.count[indices] = count_num

    def calculate_normalize_count(self):
        self.normalize_count = self.count / np.sum(self.count)
        return self.normalize_count

    def calculate_count_matrix(self):
        shape = []
        for attri in self.attributes_index:
            shape.append(self.num_categories[attri])
        self.count_matrix = np.copy(self.count).reshape(tuple(shape))
        return self.count_matrix

    # ---- General helper APIs used by update/consistency code ----
    def calculate_encode_num_general(self, attributes_index):
        categories_num = self.num_categories[attributes_index]
        categories_num = np.roll(categories_num, attributes_index.size - 1)
        categories_num[-1] = 1
        categories_num = np.flip(categories_num)
        encode_num = np.flip(np.cumprod(categories_num))
        return encode_num

    def count_records_general(self, records):
        count = np.zeros(self.num_key)
        encode_records = np.matmul(
            records[:, self.attributes_index], self.encode_num)
        encode_key, value_count = np.unique(encode_records, return_counts=True)
        indices = np.where(np.isin(np.arange(self.num_key), encode_key))[0]
        count[indices] = value_count
        return count

    def calculate_normalize_count_general(self, count):
        total = np.sum(count)
        if total == 0:
            return count
        return count / total
