# MoSuseg 数据处理仓库

MoSuseg coco数据格式准备

```
downstream
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
```

各个文件解释

```
├─coco  #   MoSuseg coco     
│  ├─annotations
│  ├─train2017
│  ├─train_lab  # 训练要用到的实例标签
│  ├─val2017
│  └─val_lab    # 测试要用到实例标签
├─datasets  
│  ├─test
│  │  ├─Binary_masks
│  │  ├─Binary_masks_instance
│  │  ├─Color_masks
│  │  ├─Color_masks_instance
│  │  ├─img
│  │  └─label
│  ├─train
│  │  ├─Binary_masks
│  │  ├─Binary_masks_instance
│  │  ├─Color_masks
│  │  ├─Color_masks_instance
│  │  ├─img
│  │  └─label
│  └─val
└─Rawdataset   # 官网下载的原始数据集
    ├─MoNuSegTestData
    │  ├─Annotations
    │  └─Tissue_Images
    └─MoNuSegTrainData
        ├─Annotations
        └─Tissue_Images
```

python执行文件解释
```
xml_to_mask.py     # 将xml文件转成mask_instance
mask_to_coco_json.py  # mask_instance转成coco格式
metric.py   # 转换和计算metric的文件
split_patches.py  # 分pathch的文件
visual_coco_json.py  # 检验可视化coco_json的效果
visual_color_mask.py  # 检验mask_instance的效果
```
