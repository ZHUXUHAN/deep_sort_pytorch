# !/usr/local/bin/python3
import os
import time
import argparse
import scipy.io
from scipy.spatial.distance import cdist
import torch
import numpy as np
import codecs
import json
from PIL import Image
import datetime
import cv2
from tqdm import tqdm

USE_TENSORRT=True
if not USE_TENSORRT:
    from bot_reid import BotReid
else:
    from bot_reid import BotReid

######################################################################
# Argument
# --------

IMAGE_SIZE=(256,128)

data_path = '/home/peif/Workspace/ReID-MGN/data/Market-1501-v15.09.15/'
gallery_path = data_path + 'bounding_box_test'
query_path = data_path + 'query'
model_path = './osnet_1.0.onnx'

# Load Data
# ---------
def findFiles(filepath,extArr,fileList):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = filepath+'/'+fi
        if os.path.isdir(fi_d):#
            findFiles(fi_d,extArr,fileList)
        else:
            ext=fi.split('.')[-1]
            if ext.lower() in extArr:
                fileList.append(fi_d)

def extractFeature(model,data):
    ff = np.empty((len(data),512), dtype=np.float32)

    for ind, img_path in enumerate(data):
        # get the inputs
        img=cv2.imread(img_path)
        img=cv2.resize(img,(IMAGE_SIZE[1],IMAGE_SIZE[0]))
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        # forward
        f = np.zeros(512, dtype=np.float32)
        for i in range(2):
            if(i==1):
                img = cv2.flip(img, 1)
            feat = model.infer(image)
            f = np.add(f, feat[0])
        fnorm = np.linalg.norm(f, ord=2)
        f = np.true_divide(f, fnorm)
        ff[ind,:] = f
    return ff

def get_id(img_path):
    camera_id = []
    labels = []
    for path in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

model = BotReid(model_path,IMAGE_SIZE, gpu_id=0)

gallery_files = []
query_files = []
findFiles(gallery_path,["jpg","jpeg","png"],gallery_files)
findFiles(query_path,["jpg","jpeg","png"],query_files)

gallery_cam,gallery_label = get_id(gallery_files)
query_cam,query_label = get_id(query_files)

t1 = time.time()
gf = extractFeature(model, gallery_files)
qf = extractFeature(model, query_files)
print("cost time:{}".format(time.time() - t1))

# Save to Matlab for check
result = {'gallery_f':gf,'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':qf,'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)

os.system('python evaluate_gpu.py')


'''
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
print(len(query_label))

for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(qf[i],query_label[i],query_cam[i],gf,gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
'''
