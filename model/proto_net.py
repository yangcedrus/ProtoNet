import paddle
import paddle.nn as nn

class ProtoNet(nn.Layer):
    def __init__(self, way_num, shot_num, query_num, use_ecu=True):
        super(ProtoNet, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.use_ecu = use_ecu


    def forward(self, query_feat, support_feat):
        t, wq, c = query_feat.shape
        _, ws, _ = support_feat.shape

        # t, wq, c
        query_feat = query_feat.reshape([t, self.way_num * self.query_num, c])
        # t, w, c
        support_feat = support_feat.reshape([t, self.way_num, self.shot_num, c])
        proto_feat = paddle.mean(support_feat, axis=2)

        # t, wq, 1, c - t, 1, w, c -> t, wq, w
        dist = -paddle.sum(
            paddle.pow(query_feat.unsqueeze(2) - proto_feat.unsqueeze(1), 2), axis=3
        )

        return dist