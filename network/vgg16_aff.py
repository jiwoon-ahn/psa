import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse

from psa.network import vgg16d
from psa.tool import pyutils


class Net(vgg16d.Net):
    def __init__(self):
        super(Net, self).__init__(fc6_dilation=4)

        self.f8_3 = nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = nn.Conv2d(512, 128, 1, bias=False)
        self.f8_5 = nn.Conv2d(1024, 256, 1, bias=False)
        self.gn8_3 = nn.modules.normalization.GroupNorm(8, 64)
        self.gn8_4 = nn.modules.normalization.GroupNorm(16, 128)
        self.gn8_5 = nn.modules.normalization.GroupNorm(32, 256)

        self.f9 = torch.nn.Conv2d(448, 448, 1, bias=False)

        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.kaiming_normal_(self.f8_5.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2]
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f8_5, self.f9]

        self.predefined_featuresize = int(448 // 8)
        self.ind_from, self.ind_to = pyutils.get_indices_of_pairs(5, (
            self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from);
        self.ind_to = torch.from_numpy(self.ind_to)

        return

    def forward(self, x, to_dense=False):

        d = super().forward_as_dict(x)

        f8_3 = F.elu(self.gn8_3(self.f8_3(d['conv4'])))
        f8_4 = F.elu(self.gn8_4(self.f8_4(d['conv5'])))
        f8_5 = F.elu(self.gn8_5(self.f8_5(d['conv5fc'])))

        x = torch.cat([f8_3, f8_4, f8_5], dim=1)
        x = F.elu(self.f9(x))

        if x.size(2) == self.predefined_featuresize and x.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            ind_from, ind_to = pyutils.get_indices_of_pairs(5, (x.size(2), x.size(3)))
            ind_from = torch.from_numpy(ind_from);
            ind_to = torch.from_numpy(ind_to)

        x = x.view(x.size(0), x.size(1), -1)

        ff = torch.index_select(x, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(x, dim=2, index=ind_to.cuda(non_blocking=True))

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

            aff_mat = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                         torch.cat([aff, torch.ones([area]), aff])).to_dense().cuda()
            return aff_mat

        else:
            return aff

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

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
