from data.mvtec3d import get_data_loader
from tqdm import tqdm
from utils.mvtec3d_util import *
import open3d as o3d
import numpy as np
import torch
from data.mvtec3d import mvtec3d_classes

    
def generate_pcd(class_name,image_size=224):
    train_loader = get_data_loader("train", class_name=class_name, img_size=image_size)
    for index,(sample,_) in tqdm(enumerate(train_loader), total =len(train_loader),desc=f'Genarate pcd for class {class_name}'): 
        #resized_organized_pc点云转化
        organized_pc_np = sample[1].squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
        #生成点云文件
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
        o3d.io.write_point_cloud("./genData/bagel/{0:03d}.ply".format(index), o3d_pc)

if __name__ == '__main__':
    classes = mvtec3d_classes()
    # for cls in classes:
    #     generate_pcd(cls)
    generate_pcd("bagel")
