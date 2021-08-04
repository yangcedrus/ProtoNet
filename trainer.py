import sys
sys.dont_write_bytecode = True

import paddle
import argparse
import os
import paddle.fluid

from model import Conv64F
from model import ProtoNet
from data import get_dataloader
from utils import accuracy, AverageMeter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_gpu', type=int, default=0, help='0 = CPU.')
parser.add_argument('--data_root', type=str, default='/data/yxs/miniImageNet--ravi')
parser.add_argument('--episode_size', type=int, default=1)
parser.add_argument('--train_episode', type=int, default=10000)
parser.add_argument('--test_episode', type=int, default=1000)
parser.add_argument('--backbone', type=str, default='ConvNet', choices=['ConvNet', 'Res12'])
parser.add_argument('--way_num', type=int, default=5)
parser.add_argument('--shot_num', type=int, default=1)
parser.add_argument('--query_num', type=int, default=15)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--use_memory', action='store_true')
config = parser.parse_args()
config.orig_imsize = -1
print(vars(config))


def _generate_local_targets(episode_size):
    local_targets = paddle.arange(config.way_num).reshape([1, -1, 1])
    local_targets = paddle.fluid.layers.expand(local_targets, [episode_size, 1, config.shot_num + config.query_num]).reshape([-1])
    return local_targets


def split_by_episode(features):
    episode_size = features.shape[0] // (config.way_num * (config.shot_num + config.query_num))
    local_labels = _generate_local_targets(episode_size).cuda().reshape([episode_size,config.way_num,config.shot_num + config.query_num])

    features = features.reshape([episode_size, config.way_num, config.shot_num + config.query_num, -1])
    support_features = features[:, :, :config.shot_num, :].reshape([episode_size, config.way_num * config.shot_num, -1])
    query_features = features[:, :, config.shot_num:, :].reshape([episode_size, config.way_num * config.query_num, -1])
    support_targets = local_labels[:, :, :config.shot_num].reshape([episode_size, -1])
    query_targets = local_labels[:, :, config.shot_num:].reshape([episode_size, -1])

    return support_features, query_features, support_targets, query_targets

def train_epoch(backbone, classifier, dataloader, optimizer, loss_fn, epoch):
    meter = AverageMeter('train', ['acc'])
    for iter_id, data in enumerate(dataloader()):

        x_data = data[0]            # 训练数据
        emb = backbone(x_data)    # 预测结果
        
        support_features, query_features, support_targets, query_targets = split_by_episode(emb)
        predicts = classifier(query_features, support_features).reshape([config.episode_size * config.way_num * config.query_num, config.way_num])

        # 计算损失
        loss = loss_fn(predicts, query_targets.reshape([-1]))

        # 计算准确率
        [acc] = accuracy(predicts, query_targets.reshape([-1]))
        meter.update('acc', acc)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 梯度清零
        optimizer.clear_grad()

        if (iter_id+1) % 100 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, iter_id+1, loss.numpy(), acc.numpy()))

    return meter.avg('acc')

@paddle.no_grad()
def val_epoch(backbone, classifier, dataloader, loss_fn, epoch):
    meter = AverageMeter('test', ['acc'])
    for iter_id, data in enumerate(dataloader()):

        x_data = data[0]          # 训练数据
        emb = backbone(x_data)    # 
        
        support_features, query_features, support_targets, query_targets = split_by_episode(emb)
        predicts = classifier(query_features, support_features).reshape([config.episode_size * config.way_num * config.query_num, config.way_num])

        # 计算损失
        loss = loss_fn(predicts, query_targets.reshape([-1]))

        # 计算准确率
        [acc] = accuracy(predicts, query_targets.reshape([-1]))
        meter.update('acc', acc)

        if (iter_id+1) % 100 == 0:
            print("epoch: {}, iter_id: {}, loss is: {}, acc is: {}".format(epoch, iter_id+1, loss.numpy(), acc.numpy()))

    return meter.avg('acc')

def main(config):
    # 用 DataLoader 实现数据加载
    train_loader = get_dataloader(config, mode='train')
    val_loader = get_dataloader(config, mode='val')
    # test_loader = get_dataloader(config, mode='test')

    # 网络搭建
    backbone = Conv64F()
    classifier = ProtoNet(config.way_num, config.shot_num, config.query_num, use_ecu=True)

    # 设置优化器
    optim = paddle.optimizer.Adam(parameters=backbone.parameters(), learning_rate=config.lr)
    # 设置损失函数
    loss_fn = paddle.nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(config.epoch):
        print("====================start training====================")
        train_acc = train_epoch(backbone, classifier, train_loader, optim, loss_fn, epoch)
        print("Avg acc:{}".format(train_acc))
        print("====================start validation====================")
        val_acc = val_epoch(backbone, classifier, val_loader, loss_fn, epoch)
        print("Avg acc:{}\tBest acc:{}".format(train_acc, best_acc))

        if val_acc>best_acc:
            best_acc = val_acc
            path = "./results/model_best.pth"
            paddle.save(backbone.state_dict(), path)

if __name__ == "__main__":
    main(config)