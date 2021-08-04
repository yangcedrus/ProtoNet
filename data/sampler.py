import numpy as np
import paddle
from paddle.fluid.layers.nn import shape
from paddle.io import Sampler
from paddle.tensor.random import randint

class CategoriesSampler(Sampler):

    def __init__(self, label_list, label_num, episode_size, episode_num, way_num,
                 image_num):
        super(CategoriesSampler, self).__init__(label_list)

        self.episode_size = episode_size
        self.episode_num = episode_num
        self.way_num = way_num
        self.image_num = image_num

        label_list = np.array(label_list)
        self.idx_list = []
        for label_idx in range(label_num):
            ind = np.argwhere(label_list == label_idx).reshape(-1)
            ind = paddle.to_tensor(ind)
            self.idx_list.append(ind)

    def __len__(self):
        return self.episode_num

    def __iter__(self):
        batch = []
        for i_batch in range(self.episode_num):
            classes = paddle.randperm(len(self.idx_list))[:self.way_num]
            for c in classes:
                idxes = self.idx_list[c.item()]
                pos = paddle.randperm(idxes.shape[0])[:self.image_num]
                batch.append(idxes[pos])
            if len(batch) == self.episode_size * self.way_num:
                # batch = paddle.stack(batch).reshape((-1))
                batch = paddle.stack(batch)
                batch = paddle.reshape(batch, shape=[-1])
                yield batch
                batch = []