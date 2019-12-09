import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from .bot_reid import BotReid


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.model = BotReid(model_path, [256, 128], gpu_id=1)

    def __call__(self, im_crops):
        fea = np.empty((len(im_crops), 512), dtype=np.float32)

        for i in range(0, len(im_crops)):
            tmpImg = cv2.resize(im_crops[i].astype(np.float32) / 255., (128, 256))

            with torch.no_grad():
                tmpFea = self.model.infer(tmpImg)
                fnorm = np.linalg.norm(tmpFea[0], ord=2)
                tmpFea = np.true_divide(tmpFea, fnorm)
                fea[i, :] = tmpFea

        return fea
