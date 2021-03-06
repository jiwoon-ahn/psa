import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse

from psa.network import resnet38d
from psa.tool import pyutils

is_cuda_available = torch.cuda.is_available()


class Net(resnet38d.Net):
    def __init__(self):
        super(Net, self).__init__()

        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f8_5 = torch.nn.Conv2d(4096, 256, 1, bias=False)

        self.f9 = torch.nn.Conv2d(448, 448, 1, bias=False)

        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.kaiming_normal_(self.f8_5.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f8_5, self.f9]

        self.predefined_featuresize = int(448 // 8)
        self.ind_from, self.ind_to = pyutils.get_indices_of_pairs(radius=5, size=(
            self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from)
        self.ind_to = torch.from_numpy(self.ind_to)

        return

    def forward(self, x, to_dense=False):

        d = super().forward_as_dict(x)

        f8_3 = F.elu(self.f8_3(d['conv4']))
        f8_4 = F.elu(self.f8_4(d['conv5']))
        f8_5 = F.elu(self.f8_5(d['conv6']))
        x = F.elu(self.f9(torch.cat([f8_3, f8_4, f8_5], dim=1)))

        if x.size(2) == self.predefined_featuresize and x.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            ind_from, ind_to = pyutils.get_indices_of_pairs(5, (x.size(2), x.size(3)))
            ind_from = torch.from_numpy(ind_from)
            ind_to = torch.from_numpy(ind_to)

        x = x.view(x.size(0), x.size(1), -1)

        if is_cuda_available:
            ind_from = ind_from.cuda(non_blocking=True)
            ind_to = ind_to.cuda(non_blocking=True)
        ff = torch.index_select(x, dim=2, index=ind_from)
        ft = torch.index_select(x, dim=2, index=ind_to)

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff = torch.exp(-torch.mean(torch.abs(ft - ff), dim=1))

        if to_dense:
            aff = aff.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = x.size(2)

            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            if is_cuda_available:
                indices_id = indices_id.cuda()
                aff = aff.cuda()
                torch_ones_cuda = torch.ones([area]).cuda()
                aff_mat = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                             torch.cat([aff, torch_ones_cuda, aff])).to_dense().cuda()

            else:
                aff_mat = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                             torch.cat([aff, torch.ones([area]), aff])).to_dense()

            return aff_mat

        else:
            return aff

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm):

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

    model = Net()
    x = torch.rand([2, 3, 448, 448])
    y = model.forward(x)

    assert len(model.ind_from) == 2496
    assert len(model.ind_to) == 84864
    assert y.shape == (2, 34, 2496)
