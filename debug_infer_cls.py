import importlib
import os.path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from omegaconf import DictConfig
from skimage.io import imsave
from torch.utils.data import DataLoader

from psa.tool import imutils
from psa.voc12 import data


@hydra.main(config_path='./conf', config_name="infer_cls")
def run_app(cfg: DictConfig) -> None:
    os.makedirs(cfg.out_cam, exist_ok=True)
    os.makedirs(cfg.out_la_crf, exist_ok=True)
    os.makedirs(cfg.out_ha_crf, exist_ok=True)
    os.makedirs(cfg.out_cam_pred, exist_ok=True)

    model = getattr(importlib.import_module(cfg.network), 'Net')()
    model.load_state_dict(torch.load(cfg.weights, map_location=torch.device('cpu')))

    model.eval()

    infer_dataset = data.VOC12ClsDatasetMSF(cfg.infer_list, voc12_root=cfg.voc12_root,
                                            cls_label_path=cfg.cls_label_path,
                                            scales=(1, 0.5, 1.5, 2.0),
                                            inter_transform=torchvision.transforms.Compose(
                                                [np.asarray,
                                                 model.normalize,
                                                 imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]
        label = label[0]

        img_path = data.get_img_path(img_name, cfg.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        cam_list = []
        # using 4 different scale with also flipped version of each.
        for i, img in enumerate(img_list):
            with torch.no_grad():
                cam = model.forward_cam(img)
                cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                # Fix the flipped image
                if i % 2 == 1:
                    cam = np.flip(cam, axis=-1)
                cam_list.append(cam)

        sum_cam = np.sum(cam_list, axis=0)
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

        cam_dict = {}
        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if cfg.out_cam is not None:
            np.save(os.path.join(cfg.out_cam, img_name + '.npy'), cam_dict)

        if cfg.out_cam_pred is not None:
            bg_score = [np.ones_like(norm_cam[0]) * 0.2]
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
            imsave(os.path.join(cfg.out_cam_pred, img_name + '.png'), pred.astype(np.uint8), check_contrast=False)

        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()

            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key + 1] = crf_score[i + 1]

            return n_crf_al

        if cfg.out_la_crf is not None:
            crf_la = _crf_with_alpha(cam_dict, cfg.low_alpha)
            np.save(os.path.join(cfg.out_la_crf, img_name + '.npy'), crf_la)

        if cfg.out_ha_crf is not None:
            crf_ha = _crf_with_alpha(cam_dict, cfg.high_alpha)
            np.save(os.path.join(cfg.out_ha_crf, img_name + '.npy'), crf_ha)

        print(iter)


if __name__ == "__main__":
    run_app()
