import paddle
import pandas as pd

def accuracy(output, target, topk=(1,)):
    """

    :param output:
    :param target:
    :param topk:
    :return:
    """
    with paddle.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.equal(target.reshape([1, -1]).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape([-1]).astype('float64').sum(0, keepdim=True)
            res.append(correct_k * (100.0 / batch_size))
        return res

class AverageMeter(object):
    """

    """

    def __init__(self, name, keys, writer=None):
        self.name = name
        self._data = pd.DataFrame(index=keys,
                                  columns=['last_value', 'total', 'counts', 'average', ])
        self.writer = writer
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        value = value.numpy().item()
        if self.writer is not None:
            tag = '{}/{}'.format(self.name, key)
            self.writer.add_scalar(tag, value)
        self._data.last_value[key] = value
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def last(self, key):
        return self._data.last_value[key]