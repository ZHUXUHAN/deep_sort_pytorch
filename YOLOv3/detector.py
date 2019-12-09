import torch
# import time
import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from darknet import Darknet

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils.utils import *
from utils.aug_for_detect import *

plt.switch_backend('agg')


class YOLOv3(object):
    def __init__(self, cfgfile, weightfile, namesfile, use_cuda=True, is_plot=False, is_xywh=False, conf_thresh=0.7,
                 nms_thresh=0.4, img_size=1024):
        # net definition
        self.net = Darknet(cfgfile)
        if weightfile.endswith(".weights"):
            # Load darknet weights
            self.net.load_darknet_weights(weightfile)
        else:
            # Load checkpoint weights
            self.net.load_state_dict(torch.load(weightfile))
        print('Loading weights from %s... Done!' % (weightfile))
        self.device = "cuda" if use_cuda else "cpu"
        self.net.eval()
        self.net.to(self.device)

        # constants
        self.size = self.net.width, self.net.height
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.use_cuda = use_cuda
        self.is_plot = is_plot
        self.is_xywh = is_xywh
        self.class_names = self.load_class_names(namesfile)
        self.image_size = img_size
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def __call__(self, ori_img):

        # forward
        input_imgs = pro_image_for_infer(ori_img, self.image_size)
        img = Variable(input_imgs.type(self.Tensor)).unsqueeze(0)
        with torch.no_grad():
            detections = self.net(img)
            detections = non_max_suppression(detections, self.conf_thresh, self.nms_thresh)[0]
            if detections is not None:
                detections = rescale_boxes(detections, self.image_size, np.array(ori_img).shape[:2])

        # plot boxes
        if self.is_plot:
            return self.plot_bbox(ori_img, detections)
        if detections is None:
            return None, None, None

        height, width = np.array(ori_img).shape[:2]
        detections = np.vstack(detections)
        bbox = np.empty_like(detections[:, :4])
        if self.is_xywh:
            # bbox x y w h
            detections[:, 0] = np.maximum(0, detections[:, 0])
            detections[:, 1] = np.maximum(0, detections[:, 1])
            detections[:, 2] = np.maximum(0, detections[:, 2])
            detections[:, 3] = np.maximum(0, detections[:, 3])

            detections[:, 0] = np.minimum(width, detections[:, 0])
            detections[:, 1] = np.minimum(height, detections[:, 1])
            detections[:, 2] = np.minimum(width, detections[:, 2])
            detections[:, 3] = np.minimum(height, detections[:, 3])

            bbox[:, 0] = (detections[:, 0] + detections[:, 2]) / 2
            bbox[:, 1] = (detections[:, 1] + detections[:, 3]) / 2
            bbox[:, 2] = detections[:, 2] - detections[:, 0] + 1
            bbox[:, 3] = detections[:, 3] - detections[:, 1] + 1

        else:
            # bbox xmin ymin xmax ymax
            bbox[:, 0] = (detections[:, 0] - detections[:, 2] / 2.0)
            bbox[:, 1] = (detections[:, 1] - detections[:, 3] / 2.0)
            bbox[:, 2] = (detections[:, 0] + detections[:, 2] / 2.0)
            bbox[:, 3] = (detections[:, 1] + detections[:, 3] / 2.0)
        cls_conf = detections[:, 5]
        cls_ids = detections[:, 6]
        return bbox, cls_conf, cls_ids

    def load_class_names(self, namesfile):
        with open(namesfile, 'r', encoding='utf8') as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names

    def plot_bbox(self, ori_img, detections):

        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        fig, ax = plt.subplots(1)
        ax.imshow(ori_img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=self.class_names[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )
        return plt


if __name__ == '__main__':
    yolo3 = YOLOv3("cfg/yolov3-voc-merge-coco.cfg",
                   "mymodel/yolov3_ckpt_78.pth", "cfg/voc.names", is_plot=True, conf_thresh=0.6)
    import os

    root = "./samples"
    files = [os.path.join(root, file) for file in os.listdir(root)]
    files.sort()
    for filename in files:
        img = Image.open(filename)
        res = yolo3(img)
        # save results
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig("./result/{}".format(os.path.basename(filename)), bbox_inches="tight", pad_inches=0.0)
        plt.close()
