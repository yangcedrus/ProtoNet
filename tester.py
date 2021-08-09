import sys
sys.dont_write_bytecode = True

import paddle
import argparse
import os
import paddle.fluid
import numpy as np
import scipy as sp
import scipy.stats

from model import Conv64F
from model import ProtoNet
from data import get_dataloader
from utils import accuracy, AverageMeter

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--n_gpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--data_root', type=str, default='/data/yxs/miniImageNet--ravi')
parser.add_argument('--checkpoint_path', type=str, default="./results/5way_5shot_model_best.pth")
parser.add_argument('--episode_size', type=int, default=1)
parser.add_argument('--train_episode', type=int, default=1000)
parser.add_argument('--test_episode', type=int, default=600)
parser.add_argument('--backbone', type=str, default='ConvNet', choices=['ConvNet', 'Res12'])
parser.add_argument('--test_way', type=int, default=5)
parser.add_argument('--shot_num', type=int, default=1)
parser.add_argument('--query_num', type=int, default=15)
parser.add_argument('--use_memory', action='store_true')
config = parser.parse_args()
config.orig_imsize = -1
print(vars(config))

if config.shot_num == 1:
    paddle.set_device("gpu:0")
else:
    paddle.set_device("gpu:1")

f = open("{}way_{}shot_test.txt".format(config.test_way, config.shot_num), 'w', buffering=1)


def mean_confidence_interval(data, confidence=0.95):
    a = [1.0 * np.array(data[i]) for i in range(len(data))]
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def _generate_local_targets(episode_size, way_num):
    local_targets = paddle.arange(way_num).reshape([1, -1, 1])
    local_targets = paddle.fluid.layers.expand(local_targets, [episode_size, 1, config.shot_num + config.query_num]).reshape([-1])
    return local_targets


def split_by_episode(features, way_num):
    episode_size = features.shape[0] // (way_num * (config.shot_num + config.query_num))
    local_labels = _generate_local_targets(episode_size, way_num).cuda().reshape([episode_size, way_num,config.shot_num + config.query_num])

    features = features.reshape([episode_size, way_num, config.shot_num + config.query_num, -1])
    support_features = features[:, :, :config.shot_num, :].reshape([episode_size, way_num * config.shot_num, -1])
    query_features = features[:, :, config.shot_num:, :].reshape([episode_size, way_num * config.query_num, -1])
    support_targets = local_labels[:, :, :config.shot_num].reshape([episode_size, -1])
    query_targets = local_labels[:, :, config.shot_num:].reshape([episode_size, -1])

    return support_features, query_features, support_targets, query_targets

@paddle.no_grad()
def val_epoch(backbone, classifier, dataloader, loss_fn, epoch):
    meter = AverageMeter('test', ['acc', 'loss'])
    acc_list = []
    for iter_id, data in enumerate(dataloader()):

        x_data = data[0]          # 训练数据
        emb = backbone(x_data)    # 
        
        support_features, query_features, support_targets, query_targets = split_by_episode(emb, config.test_way)
        predicts = classifier(query_features, support_features).reshape([config.episode_size * config.test_way * config.query_num, config.test_way])

        # 计算损失
        loss = loss_fn(predicts, query_targets.reshape([-1]))
        meter.update('loss', loss)

        # 计算准确率
        [acc] = accuracy(predicts, query_targets.reshape([-1]))
        acc_list.append(acc)
        meter.update('acc', acc)

        if (iter_id+1) % 100 == 0:
            msg = "epoch: {}, iter_id: {}, loss is: {}, acc is: {}".format(epoch, iter_id+1, meter.avg('loss'),meter.avg('acc'))
            print(msg)
            f.write(msg+"\n")

    return mean_confidence_interval(acc_list)

def main(config):
    # 用 DataLoader 实现数据加载
    test_loader = get_dataloader(config, mode='test')

    # 网络搭建
    backbone = Conv64F()
    test_classifier = ProtoNet(config.test_way, config.shot_num, config.query_num, use_ecu=True)

    # 模型加载
    checkponint = paddle.load(config.checkpoint_path)
    backbone.set_state_dict(checkponint)

    # 设置损失函数
    loss_fn = paddle.nn.CrossEntropyLoss()

    total_acc = 0
    total_h = 0
    for epoch in range(config.epoch):
        print("====================start test====================")
        f.write("====================start test===================="+"\n")
        test_acc, test_h = val_epoch(backbone, test_classifier, test_loader, loss_fn, epoch)
        print("Test acc:{}\tTest h:{}".format(test_acc, test_h))
        f.write("Test acc:{}\tTest h:{}".format(test_acc, test_h)+"\n")

        total_acc += test_acc
        total_h += test_h

    print("Mean acc:{}\tMean h:{}".format(total_acc/config.epoch, total_h/config.epoch))

if __name__ == "__main__":
    main(config)
    f.close()
