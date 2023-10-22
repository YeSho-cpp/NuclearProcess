import numpy as np
from skimage import measure
import os
from PIL import Image
import json
from pycocotools import mask as maskUtils


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred

def get_fast_aji(true, pred):
    """
    AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None, ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None, ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    # 多检
    pairwise_FP = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    # 漏检
    pairwise_FN = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    
    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter
            
            pairwise_FP[true_id - 1, pred_id - 1] = p_mask.sum() - inter
            pairwise_FN[true_id - 1, pred_id - 1] = t_mask.sum() - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()
    
    overall_FP = (pairwise_FP[paired_true, paired_pred]).sum()
    overall_FN = (pairwise_FN[paired_true, paired_pred]).sum()
    
    
    #
    paired_true = (list(paired_true + 1))  # index to instance ID
    paired_pred = (list(paired_pred + 1))
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
    
    less_pred = 0
    more_pred = 0
    
    for true_id in unpaired_true:
        less_pred += true_masks[true_id].sum()
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        more_pred += pred_masks[pred_id].sum()
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    fm = overall_union - overall_inter
    # print('\t [ana_FP = {:.4f}, ana_FN = {:.4f}, ana_less = {:.4f}, ana_more = {:.4f}]'.format((overall_FP / fm),(overall_FN / fm),(less_pred / fm),(more_pred / fm)))

    # return aji_score, overall_FP / fm, overall_FN / fm, less_pred / fm, more_pred / fm
    return aji_score
  
  
def get_dice_1(true, pred):
    """
        Traditional dice
    """
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom)


def json_to_pny(json_path,npy_path):
  # 加载 JSON 数据
  with open(json_path, "r") as f:
      data = json.load(f)
  # 按 image_id 分类数据
  instances_by_image = {}
  for item in data:
      image_id = item['image_id']
      if image_id not in instances_by_image:
          instances_by_image[image_id] = []
      instances_by_image[image_id].append(item)

  # 确保输出目录存在
  output_dir = npy_path
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  # 为每个 image_id 生成 npy 文件
  for image_id, instances in instances_by_image.items():
      instance_mask = np.zeros((250, 250), dtype=np.uint16)
      instance_id = 1  # 从1开始计数，0表示背景
      for instance in instances:
          rle = {
              'counts': instance['segmentation']['counts'],
              'size': instance['segmentation']['size']
          }
          binary_mask = maskUtils.decode(rle)
          instance_mask[binary_mask > 0] = instance_id
          instance_id += 1
      output_path = os.path.join(output_dir, f"{str(image_id).zfill(6)}.npy")
      np.save(output_path, instance_mask)

  print("Instance masks saved in .npy format!")

# 转换meric需要
def get_metric(true_path,pred_path):
    result_Dice=0
    result_AJI=0
    for pred_name in os.listdir(pred_path):
        pred=os.path.join(pred_path,pred_name)
        true_name = pred_name.split(".")[0]+".png"
        true=os.path.join(true_path,true_name)
        if true.endswith(".png"):
            true= np.array(Image.open(true))
        elif true.endswith(".npy"):
            data = np.load(true)
        if pred.endswith(".png"):
            pred=np.array(Image.open(pred))
        elif pred.endswith(".npy"):
            pred = np.load(pred)
        remap_label(true)
        true = measure.label(true)
        pred = measure.label(pred)
        result_Dice+=get_dice_1(true,pred)
        result_AJI+=get_fast_aji(true,pred)
    result_AJI=result_AJI/len(os.listdir(pred_path))
    result_Dice=result_Dice/len(os.listdir(pred_path))
    print('result_AJI = {}, result_Dice = {}'.format(
                result_AJI, result_Dice))
    
    
def main():
  # 转metric 首先将pred_json的文件转成一个个的npy文件，名字对应图片的id
  # 这个npy文件可以作为算aij和dice的pred，而ture就是label图片文件
  npy_path="out/npy" # 转npy文件的保存路径
  pred_json="" #预测产生的json文件
  label_path=""
  is_change_pny=True # 是否需要转换
  
  if is_change_pny:
    json_to_pny(pred_json,npy_path)
  get_metric(label_path,npy_path)

if __name__ == '__main__':
    main()

