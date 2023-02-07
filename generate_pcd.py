'''
Author: Yong Lei yong.lei@momenta.ai
Date: 2023-02-03 10:53:30
LastEditors: Yong Lei yong.lei@momenta.ai
LastEditTime: 2023-02-07 12:06:49
FilePath: /3D-ADS/generate_pcd.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from data.mvtec3d import get_data_loader
from tqdm import tqdm
from utils.mvtec3d_util import *
import open3d as o3d
import numpy as np
import torch
from data.mvtec3d import mvtec3d_classes
import os

    
def  train_pcd(class_name,image_size=224):
    train_loader = get_data_loader("train", class_name=class_name, img_size=image_size)
    for index,(sample,_) in tqdm(enumerate(train_loader), total =len(train_loader),desc=f'Genarate train pcd for class {class_name}'): 
        pass

def test_pcd(class_name,image_size=224):
    test_loader = get_data_loader("test", class_name=class_name, img_size=image_size)
    for index,(sample,_,_) in tqdm(enumerate(test_loader), total =len(test_loader),desc=f'Genarate test pcd for class {class_name}'): 
        pass
    





if __name__ == '__main__':
    classes = mvtec3d_classes()
    # for cls in classes:
    #     generate_pcd(cls)
    train_pcd("bagel")
    # test_pcd("bagel")
