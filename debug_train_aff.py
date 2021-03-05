import importlib

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from psa.tool import pyutils, imutils, torchutils
from psa.voc12 import data


@hydra.main(config_path='./conf', config_name="train_aff")
def run_app(cfg: DictConfig) -> None:
    pyutils.Logger(cfg.session_name + '.log')

    print(vars(cfg))

    model = getattr(importlib.import_module(cfg.network), 'Net')()

    print(model)

    train_dataset = data.VOC12AffDataset(cfg.train_list, label_la_dir=cfg.la_crf_dir,
                                         label_ha_dir=cfg.ha_crf_dir,
                                         voc12_root=cfg.voc12_root, cropsize=cfg.crop_size, radius=5,
                                         joint_transform_list=[
                                             None,
                                             None,
                                             imutils.RandomCrop(cfg.crop_size),
                                             imutils.RandomHorizontalFlip()
                                         ],
                                         img_transform_list=[
                                             transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                    hue=0.1),
                                             np.asarray,
                                             model.normalize,
                                             imutils.HWC_to_CHW
                                         ],
                                         label_transform_list=[
                                             None,
                                             None,
                                             None,
                                             imutils.AvgPool2d(8)
                                         ])

    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                   num_workers=cfg.num_workers,
                                   pin_memory=True, drop_last=True)
    max_step = len(train_dataset) // cfg.batch_size * cfg.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': cfg.lr, 'weight_decay': cfg.wt_dec},
        {'params': param_groups[1], 'lr': 2 * cfg.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * cfg.lr, 'weight_decay': cfg.wt_dec},
        {'params': param_groups[3], 'lr': 20 * cfg.lr, 'weight_decay': 0}
    ], lr=cfg.lr, weight_decay=cfg.wt_dec, max_step=max_step)

    if cfg.weights[-7:] == '.params':
        import network.resnet38d

        assert cfg.network == "network.resnet38_aff"
        weights_dict = network.resnet38d.convert_mxnet_to_torch(cfg.weights)
    elif cfg.weights[-11:] == '.caffemodel':
        import network.vgg16d

        assert cfg.network == "network.vgg16_aff"
        weights_dict = network.vgg16d.convert_caffe_to_torch(cfg.weights)
    else:
        weights_dict = torch.load(cfg.weights, map_location=torch.device('cpu'))

    model.load_state_dict(weights_dict, strict=False)
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'bg_loss', 'fg_loss', 'neg_loss', 'bg_cnt', 'fg_cnt', 'neg_cnt')

    timer = pyutils.Timer("Session started: ")

    for ep in range(cfg.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            aff = model.forward(pack[0])
            bg_label = pack[1][0]
            fg_label = pack[1][1]
            neg_label = pack[1][2]

            bg_count = torch.sum(bg_label) + 1e-5
            fg_count = torch.sum(fg_label) + 1e-5
            neg_count = torch.sum(neg_label) + 1e-5

            bg_loss = torch.sum(- bg_label * torch.log(aff + 1e-5)) / bg_count
            fg_loss = torch.sum(- fg_label * torch.log(aff + 1e-5)) / fg_count
            neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff)) / neg_count

            loss = bg_loss / 4 + fg_loss / 4 + neg_loss / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({
                'loss': loss.item(),
                'bg_loss': bg_loss.item(), 'fg_loss': fg_loss.item(), 'neg_loss': neg_loss.item(),
                'bg_cnt': bg_count.item(), 'fg_cnt': fg_count.item(), 'neg_cnt': neg_count.item()
            })

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'bg_loss', 'fg_loss', 'neg_loss'),
                      'cnt:%.0f %.0f %.0f' % avg_meter.get('bg_cnt', 'fg_cnt', 'neg_cnt'),
                      'imps:%.1f' % ((iter + 1) * cfg.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()


        else:
            print('')
            timer.reset_stage()

    torch.save(model.state_dict(), cfg.session_name + '.pth')


if __name__ == "__main__":
    run_app()
