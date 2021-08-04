import paddle
import paddle.nn as nn


class Conv64F(nn.Layer):
    """
        Four convolutional blocks network, each of which consists of a Covolutional layer,
        a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
        Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

        Input:  3 * 84 *84
        Output: 64 * 5 * 5
    """

    def __init__(self):
        super(Conv64F, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out4 = out4.reshape([x.shape[0], -1])
        return out4
