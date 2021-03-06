import argparse
import importlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from psa.tool import pyutils, imutils, torchutils
from psa.voc12 import data

is_cuda_available = torch.cuda.is_available()

if is_cuda_available:
    from torch.backends import cudnn

    cudnn.enabled = True


def validate(model, data_loader):
    print('\nvalidating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack[1]
            label = pack[2]
            if is_cuda_available:
                label = label.cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss:', val_loss_meter.pop('loss'))

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--network", default="psa/network.vgg16_cls", type=str)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--train_list", default="psa/voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="psa/voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="vgg_cls", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--voc12_root", required=True, type=str)
    args = parser.parse_args()
    if args.network == "psa.network.resnet38_cls_wildcat":
        model = getattr(importlib.import_module(args.network), 'Net')(kmax=1,
                                                                      kmin=1,
                                                                      alpha=0.7,
                                                                      num_maps=4)
    else:
        model = getattr(importlib.import_module(args.network), 'Net')()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    train_dataset = data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                         transform=transforms.Compose([
                                             imutils.RandomResizeLong(256, 512),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                    hue=0.1),
                                             np.asarray,
                                             model.normalize,
                                             imutils.RandomCrop(args.crop_size),
                                             imutils.HWC_to_CHW,
                                             torch.from_numpy
                                         ]))

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

    val_dataset = data.VOC12ClsDataset(args.val_list, voc12_root=args.voc12_root,
                                       transform=transforms.Compose([
                                           np.asarray,
                                           model.normalize,
                                           imutils.CenterCrop(500),
                                           imutils.HWC_to_CHW,
                                           torch.from_numpy
                                       ]))
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.get_parameter_groups()
    # Custom Implementation
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        assert args.network == "psa.network.resnet38_cls"
        from psa.network import resnet38d

        weights_dict = resnet38d.convert_mxnet_to_torch(args.weights)
    elif args.weights[-11:] == '.caffemodel':
        assert args.network == "psa.network.vgg16_cls"
        from psa.network import vgg16d

        weights_dict = vgg16d.convert_caffe_to_torch(args.weights)
    else:
        if is_cuda_available:
            weights_dict = torch.load(args.weights)
        else:
            weights_dict = torch.load(args.weights, map_location=torch.device('cpu'))
    if args.network == "psa.network.resnet38_cls_wildcat":
        weights_dict.pop('fc8.weight')
    model.load_state_dict(weights_dict, strict=False)
    if is_cuda_available:
        model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            img = pack[1]
            label = pack[2]
            if is_cuda_available:
                label = label.cuda(non_blocking=True)

            x = model(img)
            # Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy
            # https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.session_name + '.pth')
