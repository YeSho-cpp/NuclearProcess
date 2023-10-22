import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import morphology
import random
# 对npy文件或者实例png图片进行可视化
def display_color_mask(npy_path:str):
    if npy_path.endswith(".png"):
      data=np.array(Image.open(npy_path))
    elif npy_path.endswith(".npy"):
      data = np.load(npy_path)
    try:
        # 从.npy文件加载数据
        unique_values=np.unique(data)
        unique_values = unique_values[unique_values != 0] 
        
        # 创建颜色掩码
        color_mask = np.zeros(data.shape + (3,), dtype=np.uint8)
        for unique_value in unique_values:
          random_numbers = [random.randint(0, 256) for _ in range(3)]
          color_mask[data==unique_value]=random_numbers
        # color_mask[data != 0] = [255, 0, 0]  # 设置非零值的颜色，此处示例使用红色
        dilated_mask = morphology.binary_dilation(data != 0)
        # color_mask[dilated_mask] = [255, 255, 255]
        # 展示颜色掩码
        plt.imshow(color_mask)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"展示颜色掩码失败：{str(e)}")
        
if __name__ == '__main__':

  path="datasets/test/label/TCGA-2Z-A9J9-01A-01-TS1.png"
  display_color_mask(path)