import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class Normalize():
    def __init__(self, mean = (122.675, 116.669, 104.008)):

        self.mean = mean

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 2] - self.mean[2])
        proc_img[..., 1] = (imgarr[..., 1] - self.mean[1])
        proc_img[..., 2] = (imgarr[..., 0] - self.mean[0])

        return proc_img

class Net(nn.Module):
    def __init__(self, fc6_dilation = 1):
        super(Net, self).__init__()

        self.conv1_1 = nn.Conv2d(3,64,3,padding = 1)
        self.conv1_2 = nn.Conv2d(64,64,3,padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.conv2_1 = nn.Conv2d(64,128,3,padding = 1)
        self.conv2_2 = nn.Conv2d(128,128,3,padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.conv3_1 = nn.Conv2d(128,256,3,padding = 1)
        self.conv3_2 = nn.Conv2d(256,256,3,padding = 1)
        self.conv3_3 = nn.Conv2d(256,256,3,padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)
        self.conv4_1 = nn.Conv2d(256,512,3,padding = 1)
        self.conv4_2 = nn.Conv2d(512,512,3,padding = 1)
        self.conv4_3 = nn.Conv2d(512,512,3,padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding=1)
        self.conv5_1 = nn.Conv2d(512,512,3,padding = 2, dilation = 2)
        self.conv5_2 = nn.Conv2d(512,512,3,padding = 2, dilation = 2)
        self.conv5_3 = nn.Conv2d(512,512,3,padding = 2, dilation = 2)
        self.pool5 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size = 3, stride = 1, padding=1)

        self.fc6 = nn.Conv2d(512,1024, 3, padding = fc6_dilation, dilation = fc6_dilation)

        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Conv2d(1024,1024,1)

        self.normalize = Normalize()

        return

    def forward(self, x):
        return self.forward_as_dict(x)['conv5fc']

    def forward_as_dict(self, x):

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        conv4 = x

        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        conv5 = x

        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        x = F.relu(self.fc7(x))

        conv5fc = x

        return dict({'conv4': conv4, 'conv5': conv5, 'conv5fc': conv5fc})

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):

                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

def convert_caffe_to_torch(caffemodel_path, prototxt_path='network/vgg16_20M.prototxt'):
    import caffe

    caffe_model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

    dict = {}
    for caffe_name in list(caffe_model.params.keys()):
        dict[caffe_name + '.weight'] = torch.from_numpy(caffe_model.params[caffe_name][0].data)
        dict[caffe_name + '.bias'] = torch.from_numpy(caffe_model.params[caffe_name][1].data)

    return dict




