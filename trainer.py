import paddle
from model import Conv64F
from model import ProtoNet

# dataset与mnist的定义与第一部分内容一致

# 用 DataLoader 实现数据加载
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 网络搭建
backbone = Conv64F()
classifier = ProtoNet(way_num, shot_num, query_num, use_ecu=True)

# 设置迭代次数
epochs = 5

# 设置优化器
optim = paddle.optimizer.Adam(parameters=backbone.parameters())
# 设置损失函数
loss_fn = paddle.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader()):

        x_data = data[0]            # 训练数据
        y_data = data[1]            # 训练数据标签
        predicts = mnist(x_data)    # 预测结果

        # 计算损失 等价于 prepare 中loss的设置
        loss = loss_fn(predicts, y_data)

        # 计算准确率 等价于 prepare 中metrics的设置
        acc = paddle.metric.accuracy(predicts, y_data)

        # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中

        # 反向传播
        loss.backward()

        if (batch_id+1) % 900 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id+1, loss.numpy(), acc.numpy()))

        # 更新参数
        optim.step()

        # 梯度清零
        optim.clear_grad()
##保存模型，会生成*.pdmodel、*.pdiparams、*.pdiparams.info三个模型文件
path='./mnist/inference_model'
paddle.jit.save(layer=mnist,path=path)