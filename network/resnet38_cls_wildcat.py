import torch
import torch.nn as nn
import torch.nn.functional as F

from psa.network import resnet38d
from psa.network.pooling import ClassWisePool, WildcatPool2d


class Net(resnet38d.Net):
    def __init__(self, kmax=1, kmin=None, alpha=1, num_maps=1):
        super().__init__()

        self.dropout7 = torch.nn.Dropout2d(0.5)
        num_classes = num_maps * 20

        self.fc8 = nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]
        pooling = nn.Sequential()
        pooling.add_module('class_wise', ClassWisePool(num_maps))
        pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
        self.pooling = pooling

    def forward(self, x):
        x = super().forward(x)
        x = self.dropout7(x)
        x = self.fc8(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)

        return x

    def forward_cam(self, x):
        x = super().forward(x)

        x = self.pooling.class_wise(x)
        x = F.conv2d(x, self.fc8.weight)
        ## Doing this to not use bias values
        x = F.relu(x)

        return x

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups


if __name__ == '__main__':
    from torchsummary import summary

    summary(Net(), input_size=(3, 448, 448))

    model = Net(num_maps=4, alpha=0.7)
    x = torch.rand([2, 3, 448, 448])
    y = model.forward(x)
    model.eval()
    cam = model.forward_cam(x)
    assert y.shape == (2, 20)
    assert cam.shape == (2, 20, 56, 56)
