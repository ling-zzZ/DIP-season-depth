# -*- coding: utf-8 -*-
"""
Created on Thu May 19 22:18:44 2022

@author: ling-zzZ
"""

import os
import sys
from dpt.models import DPTDepthModel
import cv2
from PIL import Image
import torch
import argparse
import numpy as np

def getresults(left, func):

    in_h, in_w = left.shape[:2]

    left_img = cv2.resize(left, (outw, outh), interpolation=cv2.INTER_CUBIC)
    left_img = cv2.cvtColor(left_img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    left_img = (left_img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    left_img = left_img.transpose(2, 0, 1).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_left = torch.from_numpy(left_img).to(device).unsqueeze(0)

    with torch.no_grad():
        output = func(batch_left)

    pred_depth = output.squeeze().cpu().data.numpy()
    if np.min(pred_depth) < 0:
        pred_depth = (pred_depth -np.min(pred_depth))/np.max(pred_depth) * 65535.0
    else:
        pred_depth = pred_depth /np.max(pred_depth) * 65535.0
    pred_depth = cv2.resize(pred_depth, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    res = pred_depth.astype(np.uint16)

    return res


def load_model(model_path):
    net = DPTDepthModel(
            img_size = [outh, outw],
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
            invert=True,
        )
    torch.cuda.empty_cache()
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.to(device)

    return net

def main():
    parser = argparse.ArgumentParser(description="DPT")

    parser.add_argument('--test_data', default='./img/img_03170_c0_1303398750047535us.jpg')
    parser.add_argument('--model_set', default='dpt_season_weight.pkl')
    parser.add_argument('--size', default='576x768')
    parser.add_argument('--num', default='10000', type=int)
    parser.add_argument('--name', default='', help="using for result dir")

    args = parser.parse_args()
    global outh
    global outw
    outh, outw = [int(e) for e in args.size.split('x')]
    net = load_model(args.model_set)
    img_raw = cv2.imread(args.test_data, cv2.IMREAD_COLOR)
    depth = getresults(img_raw, net)
#    depth = Image.fromarray(depth)
#    depth.save('test_img_depth.png')     
    cv2.imwrite('./test_img_depth.png', depth)

if __name__ == "__main__":
    main()