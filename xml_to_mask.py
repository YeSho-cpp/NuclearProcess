import matplotlib.pyplot as plt 
import xml.etree.ElementTree as ET
from PIL import Image
from argparse import ArgumentParser
import argparse
from skimage import draw
import numpy as np
from os import listdir
from os.path import isfile, join
import os

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def poly2mask(vertex_row_coords, vertex_col_coords, shape,value):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, int)
    mask[fill_row_coords, fill_col_coords] = value
    return mask

def he_to_binary_mask(root_img,root_ann,color_root,binary_root,binary_root_instance,color_root_instance,label_root,\
    filename=None,plot=False,instance=True,):
    """
    Convert XML annotation file to a mask image. 
    root_img : root of the TIF images from the MoNuSegDataset
    root_ann : root of the XML annotation from the MoNuSegDataset
    color_root: Path to the save directory of the color mask 
    binary_root : Path to the save directory of the binary mask. 
    file_name : name of the file to Convert 
    plot : Plot the generated mask during the conversion
    """
    im_file = join(root_img, filename+'.tif')
    xml_file =  join(root_ann, filename+'.xml')
    tree = ET.parse(xml_file)
    xDoc = tree.getroot()
    regions = xDoc.iter('Region')# get a list of all the region tags
    array_xy = []
    instance=True
    for i,region in enumerate(regions): # Region = nuclei 
        #Region = Regions.item(regioni)    # for each region tag

        #get a list of all the vertexes (which are in order)
        verticies = region.iter('Vertex')
        l_verticies = len(list(region.iter('Vertex')))
        #xy(i + 1).lvalue = zeros(l_verticies, 2)    #allocate space for them
        xy = []
        for vertexi,vertex in enumerate(region.iter('Vertex')):        #iterate through all verticies
            #get the x value of that vertex
            x = float(vertex.attrib['X'])
            y = float(vertex.attrib['Y'])


            #get the y value of that vertex
            
            xy.append([x, y])        # finally save them into the array
        array_xy.append(xy)
    print('LEN DE REGION',i)
    array_xy = np.array(array_xy)  
    im = Image.open(im_file)
    ncol,nrow = im.size
    binary_mask = np.zeros((nrow,ncol))
    color_mask = np.zeros((3,nrow, ncol))

    #mask_final = [];
    print('LEN DE ARRAY_XY',len(array_xy))
    for i,r in enumerate(array_xy):    #for each region
        #print('Processing object # %d \\n', i)
        smaller_x = np.array(r)[:,0] 
        #print(smaller_x)
        smaller_y = np.array(r)[:,1]
        #print(smaller_y)
        #make a mask and add it to the current mask
        #this addition makes it obvious when more than 1 layer overlap each
        #other, can be changed to simply an OR depending on application.
        if instance:
            value = i+1
        else:
            value = 1
        polygon = poly2mask(smaller_x, smaller_y, (nrow, ncol),value=value) # i+1 -> je ne veux pas de 0
        #polygon = polygon*1 # Convert a bool array into 1 and 0 array 
        binary_mask = binary_mask +  np.where((polygon > 0) & ( binary_mask > 0),0,polygon)    # Where overlap -> 0 
        # binary_mask = binary_mask + i @ (1 - min(1, np.amin(binary_mask))) * polygon
        color_mask = color_mask + np.stack((np.random.rand() * polygon, np.random.rand()* polygon, np.random.rand() * polygon))
        #binary mask for all objects
        #imshow(ditance_transform)

    

    binary_mask = binary_mask.T
    binary_mask = binary_mask.astype(int)
    color_mask = color_mask.T
    if plot: 
        plt.subplot(2, 2, 1)
        plt.imshow(im)

        plt.subplot(2, 2, 2)
        plt.imshow(binary_mask)

        plt.subplot(2, 2, 3)
        plt.imshow(im)

        plt.subplot(2, 2, 4)
        plt.imshow(color_mask)

        plt.show()
    
    print('Saving the generated mask',filename,'in',binary_root)
    Save = True
    if Save:
        if not os.path.exists(binary_root):
            os.makedirs(binary_root)

        if not os.path.exists(color_root):
            os.makedirs(color_root)
        if not os.path.exists(binary_root_instance):
            os.makedirs(binary_root_instance)

        if not os.path.exists(color_root_instance):
            os.makedirs(color_root_instance)
        if not os.path.exists(label_root):
            os.makedirs(label_root)
        #print(np.unique(binary_mask))
        
        if instance : 
            np.save(join(binary_root_instance,filename+'.npy'),binary_mask)
            np.save(join(color_root_instance,filename+'.npy'),color_mask)
            image = Image.fromarray(binary_mask)
            image.save(join(label_root,filename+'.png'))
            print('Successful Saving')
        else:

            im_mask = Image.fromarray((binary_mask).astype(np.uint8))
            im_color_mask = Image.fromarray((color_mask).astype(np.uint8))
            im_mask.save(join(binary_root,filename+'.png'))
            im_color_mask.save(join(color_root,filename+'.png'))
            print('Successful Saving')

def save_img(root_img,imag_root):
    if not os.path.exists(imag_root):
        os.mkdir(imag_root)
    for filename in os.listdir(root_img):
        if filename.endswith(".tif"):
            # 构建输入和输出文件的完整路径
            input_path = os.path.join(root_img, filename)
            output_path = os.path.join(imag_root, f"{os.path.splitext(filename)[0]}.png")
            # 打开.tif文件
            image = Image.open(input_path)
            # 将图像保存为.png文件
            image.save(output_path, 'PNG')
            print(f"成功将{input_path}转换为{output_path}")
def main(): 
    root = 'Rawdataset/MoNuSegTrainData' # root代表你要处理的训练还是测试数据
    root_img = join(root,'Tissue_Images')
    root_ann = join(root,'Annotations')
    trian_or_test="train" # 你要处理的训练还是测试数据
    if trian_or_test=="train":
        data_root="datasets/train"
    elif trian_or_test=="test":
        data_root="datasets/test"
    color_root = join(data_root,'Color_masks')
    binary_root = join(data_root,'Binary_masks')
    binary_root_instance = join(data_root,'Binary_masks_instance') # 这个是实例mask的保存位置
    color_root_instance = join(data_root,'Color_masks_instance') 
    imag_root=join(data_root,'img')
    label_root=join(data_root,'label')
    save_img(root_img,imag_root)
    files = [f for f in listdir(root_img) if isfile(join(root_img, f))]
    for f in files: 
        f = f[:-4] # Delete the extension
        print('Mask image conversion of the file',f)
        he_to_binary_mask(root_img,root_ann,color_root,binary_root,label_root=label_root,filename=f,plot=False,instance=True,\
            binary_root_instance=binary_root_instance,color_root_instance=color_root_instance)
        print('Conversion done and saved in the',binary_root,'directory')
    print('All the XML annotations converted and successfully saved')

if __name__ == '__main__':
    main()
