import importlib
import os.path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig
from skimage.io import imsave
from torch.utils.data import DataLoader

from psa.tool import imutils
from psa.voc12 import data


@hydra.main(config_path='./conf', config_name="infer_aff")
def run_app(cfg: DictConfig) -> None:
    os.makedirs(cfg.out_rw, exist_ok=True)

    model = getattr(importlib.import_module(cfg.network), 'Net')()

    model.load_state_dict(torch.load(cfg.weights, map_location=torch.device('cpu')))

    model.eval()

    infer_dataset = data.VOC12ImageDataset(cfg.infer_list, voc12_root=cfg.voc12_root,
                                           transform=torchvision.transforms.Compose(
                                               [np.asarray,
                                                model.normalize,
                                                imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    for iter, (name, img) in enumerate(infer_data_loader):

        name = name[0]
        print(iter)

        orig_shape = img.shape
        padded_size = (int(np.ceil(img.shape[2] / 8) * 8), int(np.ceil(img.shape[3] / 8) * 8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2] / 8))
        dwidth = int(np.ceil(img.shape[3] / 8))

        cam = np.load(os.path.join(cfg.cam_dir, name + '.npy'), allow_pickle=True).item()

        cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
        for k, v in cam.items():
            cam_full_arr[k + 1] = v
        cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False)) ** cfg.alpha
        cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

        with torch.no_grad():
            aff_mat = torch.pow(model.forward(img, True), cfg.beta)

            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            for _ in range(cfg.logt):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            cam_full_arr = torch.from_numpy(cam_full_arr)
            cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)

            cam_vec = cam_full_arr.view(21, -1)
            cam_rw = torch.matmul(cam_vec, trans_mat)
            cam_rw = cam_rw.view(1, 21, dheight, dwidth)

            cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)
            _, cam_rw_pred = torch.max(cam_rw, 1)

            res = np.uint8(cam_rw_pred.cpu().data[0])[:orig_shape[2], :orig_shape[3]]

            imsave(os.path.join(cfg.out_rw, name + '.png'), res)


if __name__ == "__main__":
    run_app()
