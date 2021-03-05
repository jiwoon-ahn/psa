import importlib

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from psa.tool import imutils
from psa.tool import pyutils, torchutils
from psa.train_cls import validate
from psa.voc12 import data


@hydra.main(config_path='./conf', config_name="train_cls")
def run_app(cfg: DictConfig) -> None:
    model = getattr(importlib.import_module(cfg.network), 'Net')()

    pyutils.Logger(cfg.session_name + '.log')

    print(vars(cfg))

    train_dataset = data.VOC12ClsDataset(cfg.train_list, voc12_root=cfg.voc12_root,
                                         cls_label_path=cfg.cls_label_path,
                                         transform=transforms.Compose([
                                             imutils.RandomResizeLong(256, 512),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                    hue=0.1),
                                             np.asarray,
                                             model.normalize,
                                             imutils.RandomCrop(cfg.crop_size),
                                             imutils.HWC_to_CHW,
                                             torch.from_numpy
                                         ]))

    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // cfg.batch_size) * cfg.max_epoches

    val_dataset = data.VOC12ClsDataset(cfg.val_list, voc12_root=cfg.voc12_root,
                                       cls_label_path=cfg.cls_label_path,
                                       transform=transforms.Compose([
                                           np.asarray,
                                           model.normalize,
                                           imutils.CenterCrop(500),
                                           imutils.HWC_to_CHW,
                                           torch.from_numpy
                                       ]))
    val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                                 shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.get_parameter_groups()
    # Custom Implementation
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': cfg.lr, 'weight_decay': cfg.wt_dec},
        {'params': param_groups[1], 'lr': 2 * cfg.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * cfg.lr, 'weight_decay': cfg.wt_dec},
        {'params': param_groups[3], 'lr': 20 * cfg.lr, 'weight_decay': 0}
    ], lr=cfg.lr, weight_decay=cfg.wt_dec, max_step=max_step)

    if cfg.weights[-7:] == '.params':
        assert cfg.network == "network.resnet38_cls"
        import network.resnet38d

        weights_dict = network.resnet38d.convert_mxnet_to_torch(cfg.weights)
    elif cfg.weights[-11:] == '.caffemodel':
        assert cfg.network == "network.vgg16_cls"
        import network.vgg16d

        weights_dict = network.vgg16d.convert_caffe_to_torch(cfg.weights)
    else:
        weights_dict = torch.load(cfg.weights, map_location=torch.device('cpu'))

    model.load_state_dict(weights_dict, strict=False)
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    for ep in range(cfg.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            img = pack[1]
            label = pack[2]

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
                      'imps:%.1f' % ((iter + 1) * cfg.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.state_dict(), cfg.session_name + '.pth')


if __name__ == "__main__":
    run_app()
