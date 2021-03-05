import argparse
import importlib
import os.path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from skimage.io import imsave
from torch.utils.data import DataLoader

from psa.tool import imutils, pyutils
from psa.voc12 import data

is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    from torch.backends import cudnn

    cudnn.enabled = True
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.vgg16_cls", type=str)
    parser.add_argument("--infer_list", default="psa/voc12/val.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", required=True, type=str)
    parser.add_argument("--low_alpha", default=4, type=int)
    parser.add_argument("--high_alpha", default=32, type=int)
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--out_la_crf", default=None, type=str)
    parser.add_argument("--out_ha_crf", default=None, type=str)
    parser.add_argument("--out_cam_pred", default=None, type=str)

    args = parser.parse_args()
    os.makedirs(args.out_cam, exist_ok=True)
    os.makedirs(args.out_la_crf, exist_ok=True)
    os.makedirs(args.out_ha_crf, exist_ok=True)
    os.makedirs(args.out_cam_pred, exist_ok=True)

    model = getattr(importlib.import_module(args.network), 'Net')()
    if is_cuda_available:
        model.load_state_dict(torch.load(args.weights))
    else:
        model.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))

    model.eval()
    if is_cuda_available:
        model.cuda()

    infer_dataset = data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                            scales=(1, 0.5, 1.5, 2.0),
                                            inter_transform=torchvision.transforms.Compose(
                                                [np.asarray,
                                                 model.normalize,
                                                 imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    n_gpus = 0
    if is_cuda_available:
        n_gpus = torch.cuda.device_count()
        model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]
        label = label[0]

        img_path = data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]


        def _work(i, img):
            with torch.no_grad():
                if n_gpus:
                    with torch.cuda.device(i % n_gpus):
                        img = img.cuda()
                        cam = model_replicas[i % n_gpus].forward_cam(img)
                        cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                        cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                        if i % 2 == 1:
                            cam = np.flip(cam, axis=-1)
                        return cam
                else:
                    cam = model.forward_cam(img)
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam


        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=12, prefetch_size=0, processes=args.num_workers)

        cam_list = thread_pool.pop_results()

        sum_cam = np.sum(cam_list, axis=0)
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

        cam_dict = {}
        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        if args.out_cam_pred is not None:
            bg_score = [np.ones_like(norm_cam[0]) * 0.2]
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
            imsave(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))


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


        if args.out_la_crf is not None:
            crf_la = _crf_with_alpha(cam_dict, args.low_alpha)
            np.save(os.path.join(args.out_la_crf, img_name + '.npy'), crf_la)

        if args.out_ha_crf is not None:
            crf_ha = _crf_with_alpha(cam_dict, args.high_alpha)
            np.save(os.path.join(args.out_ha_crf, img_name + '.npy'), crf_ha)

        print(iter)
