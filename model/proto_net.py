import paddle
import paddle.nn as nn

class ProtoNet(nn.Layer):
    """
    原型网络分类器
    """
    def __init__(self, way_num, shot_num, query_num, use_ecu=True):
        super(ProtoNet, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.use_ecu = use_ecu


    def forward(self, query_feat, support_feat):
        t, wq, c = query_feat.shape
        _, ws, _ = support_feat.shape

        # 处理query set特征
        query_feat = query_feat.reshape([t, self.way_num * self.query_num, c])
        # 处理support set特征
        support_feat = support_feat.reshape([t, self.way_num, self.shot_num, c])
        # 计算原型, Compute prototype from support examples，对应论文公式(1)
        proto_feat = paddle.mean(support_feat, axis=2)

        # 计算query set与每个原型之间距离，使用欧式距离
        dist = -paddle.sum(
            paddle.pow(query_feat.unsqueeze(2) - proto_feat.unsqueeze(1), 2), axis=3
        )

        return dist