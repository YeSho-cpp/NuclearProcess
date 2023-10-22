import json
import copy
import numpy as np
from pycocotools import mask
import cv2
import os
import sys
from PIL import Image
from skimage import measure
import shutil
import random
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from datetime import datetime
import torch
import xml.etree.ElementTree as ET
from skimage import measure
import json
from pycocotools import mask as maskUtils
import imageio


# 生成mask
def maskToanno(ground_truth_mask,ann_count,segmentation_id):
  unique_values = np.unique(ground_truth_mask)  # 获取图像中的唯一非零值
  unique_values = unique_values[unique_values != 0] 
  annotations = []  # 一幅图片所有的annotations
  for value in unique_values:
    mask = (ground_truth_mask == value).astype(np.uint8)  # 创建新的二值mask，仅包含当前value对应的部分
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # 根据二值mask找到轮廓
    
    for contour in contours:
      annotation = {
          "segmentation": [contour.flatten().tolist()],
          "area": cv2.contourArea(contour),
          "iscrowd": 0,
          "image_id": ann_count,
          "bbox": list(cv2.boundingRect(contour)),
          "category_id": 1,
          "id": segmentation_id
      }
      segmentation_id += 1
      annotations.append(annotation)
  return annotations,segmentation_id
def get_annotations(input_filename,block_mask_path):
  block_mask_image_files = os.listdir(block_mask_path)
  coco_json=dict()
  coco_json['annotations']=[]
  coco_annos=coco_json['annotations']
  segmentation_id=1
  for mask_img in block_mask_image_files:
    annCount=int(mask_img.split('.')[0])
    block_im=np.asarray(Image.open(os.path.join(block_mask_path,mask_img)))
    block_anno,segmentation_id=maskToanno(block_im,annCount,segmentation_id)
    coco_annos.extend(block_anno)
  # coco_json["annotations"]=coco_annos
  with open(input_filename, "w") as f:
    json.dump(coco_json, f)


# 将图片img属性的补全
def format_filename(img_path:str,json_path:str):
  with open(json_path, 'r') as file:
    data = json.load(file)
  data['images']=[]
  images=data['images']
  for img_name in os.listdir(img_path):
      img=os.path.join(img_path, img_name)
      my_dict = {}
      my_dict['file_name'] = img_name
      my_dict['height'] = Image.open(img).size[0]
      my_dict['width'] = Image.open(img).size[1]
      my_dict['id'] = int(img_name.split('.')[0])
      images.append(my_dict)
  # 写回JSON文件
  with open(json_path, 'w') as file:
    json.dump(data, file, indent=4)

# 加入信息
def add_info(json_path:str):
  with open(json_path, 'r') as file:
    data = json.load(file)
  data["info"]={"description":"COCO 2017 Dataset","url":"http://cocodataset.org","version":"1.0","year":2017,"contributor":"COCO Consortium","date_created":"2017/09/01"}
  data["licenses"]=[{"url":"http://creativecommons.org/licenses/by-nc-sa/2.0/","id":1,"name":"Attribution-NonCommercial-ShareAlike License"},{"url":"http://creativecommons.org/licenses/by-nc/2.0/","id":2,"name":"Attribution-NonCommercial License"},{"url":"http://creativecommons.org/licenses/by-nc-nd/2.0/","id":3,"name":"Attribution-NonCommercial-NoDerivs License"},{"url":"http://creativecommons.org/licenses/by/2.0/","id":4,"name":"Attribution License"},{"url":"http://creativecommons.org/licenses/by-sa/2.0/","id":5,"name":"Attribution-ShareAlike License"},{"url":"http://creativecommons.org/licenses/by-nd/2.0/","id":6,"name":"Attribution-NoDerivs License"},{"url":"http://flickr.com/commons/usage/","id":7,"name":"No known copyright restrictions"},{"url":"http://www.usa.gov/copyright.shtml","id":8,"name":"United States Government Work"}]
  data["categories"]=[{"supercategory":"person","id":1,"name":"person"},{"supercategory":"vehicle","id":2,"name":"bicycle"},{"supercategory":"vehicle","id":3,"name":"car"},{"supercategory":"vehicle","id":4,"name":"motorcycle"},{"supercategory":"vehicle","id":5,"name":"airplane"},{"supercategory":"vehicle","id":6,"name":"bus"},{"supercategory":"vehicle","id":7,"name":"train"},{"supercategory":"vehicle","id":8,"name":"truck"},{"supercategory":"vehicle","id":9,"name":"boat"},{"supercategory":"outdoor","id":10,"name":"traffic light"},{"supercategory":"outdoor","id":11,"name":"fire hydrant"},{"supercategory":"outdoor","id":13,"name":"stop sign"},{"supercategory":"outdoor","id":14,"name":"parking meter"},{"supercategory":"outdoor","id":15,"name":"bench"},{"supercategory":"animal","id":16,"name":"bird"},{"supercategory":"animal","id":17,"name":"cat"},{"supercategory":"animal","id":18,"name":"dog"},{"supercategory":"animal","id":19,"name":"horse"},{"supercategory":"animal","id":20,"name":"sheep"},{"supercategory":"animal","id":21,"name":"cow"},{"supercategory":"animal","id":22,"name":"elephant"},{"supercategory":"animal","id":23,"name":"bear"},{"supercategory":"animal","id":24,"name":"zebra"},{"supercategory":"animal","id":25,"name":"giraffe"},{"supercategory":"accessory","id":27,"name":"backpack"},{"supercategory":"accessory","id":28,"name":"umbrella"},{"supercategory":"accessory","id":31,"name":"handbag"},{"supercategory":"accessory","id":32,"name":"tie"},{"supercategory":"accessory","id":33,"name":"suitcase"},{"supercategory":"sports","id":34,"name":"frisbee"},{"supercategory":"sports","id":35,"name":"skis"},{"supercategory":"sports","id":36,"name":"snowboard"},{"supercategory":"sports","id":37,"name":"sports ball"},{"supercategory":"sports","id":38,"name":"kite"},{"supercategory":"sports","id":39,"name":"baseball bat"},{"supercategory":"sports","id":40,"name":"baseball glove"},{"supercategory":"sports","id":41,"name":"skateboard"},{"supercategory":"sports","id":42,"name":"surfboard"},{"supercategory":"sports","id":43,"name":"tennis racket"},{"supercategory":"kitchen","id":44,"name":"bottle"},{"supercategory":"kitchen","id":46,"name":"wine glass"},{"supercategory":"kitchen","id":47,"name":"cup"},{"supercategory":"kitchen","id":48,"name":"fork"},{"supercategory":"kitchen","id":49,"name":"knife"},{"supercategory":"kitchen","id":50,"name":"spoon"},{"supercategory":"kitchen","id":51,"name":"bowl"},{"supercategory":"food","id":52,"name":"banana"},{"supercategory":"food","id":53,"name":"apple"},{"supercategory":"food","id":54,"name":"sandwich"},{"supercategory":"food","id":55,"name":"orange"},{"supercategory":"food","id":56,"name":"broccoli"},{"supercategory":"food","id":57,"name":"carrot"},{"supercategory":"food","id":58,"name":"hot dog"},{"supercategory":"food","id":59,"name":"pizza"},{"supercategory":"food","id":60,"name":"donut"},{"supercategory":"food","id":61,"name":"cake"},{"supercategory":"furniture","id":62,"name":"chair"},{"supercategory":"furniture","id":63,"name":"couch"},{"supercategory":"furniture","id":64,"name":"potted plant"},{"supercategory":"furniture","id":65,"name":"bed"},{"supercategory":"furniture","id":67,"name":"dining table"},{"supercategory":"furniture","id":70,"name":"toilet"},{"supercategory":"electronic","id":72,"name":"tv"},{"supercategory":"electronic","id":73,"name":"laptop"},{"supercategory":"electronic","id":74,"name":"mouse"},{"supercategory":"electronic","id":75,"name":"remote"},{"supercategory":"electronic","id":76,"name":"keyboard"},{"supercategory":"electronic","id":77,"name":"cell phone"},{"supercategory":"appliance","id":78,"name":"microwave"},{"supercategory":"appliance","id":79,"name":"oven"},{"supercategory":"appliance","id":80,"name":"toaster"},{"supercategory":"appliance","id":81,"name":"sink"},{"supercategory":"appliance","id":82,"name":"refrigerator"},{"supercategory":"indoor","id":84,"name":"book"},{"supercategory":"indoor","id":85,"name":"clock"},{"supercategory":"indoor","id":86,"name":"vase"},{"supercategory":"indoor","id":87,"name":"scissors"},{"supercategory":"indoor","id":88,"name":"teddy bear"},{"supercategory":"indoor","id":89,"name":"hair drier"},{"supercategory":"indoor","id":90,"name":"toothbrush"}]
  
    # 写回JSON文件
  with open(json_path, 'w') as file:
    json.dump(data, file, indent=4)

def post_check(json_path):
  # Load the JSON file
  with open(json_path, "r") as val_json:
      json_object = json.load(val_json)

  # Iterate over each annotation and modify segmentations of length 4
  for instance in json_object["annotations"]:
      if len(instance["segmentation"][0]) == 2:
        segment = instance["segmentation"][0]
        instance["segmentation"][0] = [segment[0], segment[1], segment[0], segment[1]]
      if len(instance["segmentation"][0]) == 4:
          segment = instance["segmentation"][0]
          instance["segmentation"][0] = [segment[0], segment[1], segment[0], segment[1], segment[2], segment[3]]
  # Save the modified JSON object back to the file
  with open(json_path, "w") as val_json:
      json.dump(json_object, val_json, indent=4)

  print("JSON file has been modified and saved.")
def main():
  json_path="coco/annotations/instances_val2017.json" # 保存coco_json的位置
  label_path="coco/val_lab"  # 产生json标注所需要的maks_instance图片位置
  img_path="coco/val2017" 
  get_annotations(json_path,label_path) # 
  format_filename(img_path,json_path)
  add_info(json_path)
  post_check(json_path) # 后检查防止一个annotations的点太少
if __name__ == '__main__':
    main()