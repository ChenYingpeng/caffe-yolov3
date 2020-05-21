#Company:	Synthesis 
#Author: 	Chen
#Date:	2020/04/26	
 
"""
COCO-Style Evaluations

put images here datasets/your_project_name/annotations/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import argparse
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-json', type=str, default='/home/chen/data/coco2017/annotations/instances_val2017.json', help='coco val2017 annotations json files')
    ap.add_argument('--pred-json', type=str, default='results/darknet_yolov3_coco_results.json', help='pred coco val2017 annotations json files')
    args = ap.parse_args()
    print(args)

    pred_json_path = args.pred_json

    MAX_IMAGES = 10000
    coco_gt = COCO(args.gt_json)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    eval(coco_gt, image_ids, pred_json_path)
