
import imageio
import os
# 分patch
def split_patches(data_dir, save_dir,patch_size,is_chage_name,post_fix=None):
    import math
    """ split large image into small patches """
    image_list = os.listdir(data_dir)
    for image_name in image_list:
        name = image_name.split('.')[0]
        if post_fix and name[-len(post_fix):] != post_fix:
            continue
        image_path = os.path.join(data_dir, image_name)
        image = imageio.v2.imread(image_path)
        seg_imgs = []
        # split into 16 patches of size 250x250
        h, w = image.shape[0], image.shape[1]
        h_overlap = math.ceil((4 * patch_size - h) / 3)
        w_overlap = math.ceil((4 * patch_size - w) / 3)
        for i in range(0, h-patch_size+1, patch_size-h_overlap):
            for j in range(0, w-patch_size+1, patch_size-w_overlap):
                if len(image.shape) == 3:
                    patch = image[i:i+patch_size, j:j+patch_size, :]
                else:
                    patch = image[i:i + patch_size, j:j + patch_size]
                seg_imgs.append(patch)

        for k in range(len(seg_imgs)):
            if post_fix:
                imageio.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(post_fix)-1], k, post_fix), seg_imgs[k])
            else:
                imageio.imsave('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])
    if is_chage_name:
      folder_path = save_dir
      extensions = ['.jpg', '.png']
      new_name = '{}{}'
      counter = 1
      for filename in sorted(os.listdir(folder_path)):
          if filename.endswith(tuple(extensions)):
              extension = os.path.splitext(filename)[1]
              new_filename = new_name.format(str(counter).zfill(6), extension)
              old_filepath = os.path.join(folder_path, filename)
              new_filepath = os.path.join(folder_path, new_filename)
              os.rename(old_filepath, new_filepath)
              counter += 1
                
if __name__ == '__main__':
  before_split_path="datasets/test/label" # 分之前的文件路径
  after_split_path="coco/val_lab" # 分完的path保存的路径
  patch_size = 250  #todo 这个是分path的大小
  is_chage_name=True # 是否将TCGA-xxx 这种改成0000x.png 方便做出coco_json
  split_patches(before_split_path, after_split_path,patch_size,is_chage_name)