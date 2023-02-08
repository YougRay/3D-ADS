import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
from utils.mvtec3d_util import *
import open3d as o3d
import numpy as np

DATASETS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '/content/drive/MyDrive/', 'dataset'))


# 选择数据集中需要的类
def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]

#生成样本读取路径、降采样rgb图片
class MVTec3D(Dataset):

    def __init__(self, split, class_name, img_size):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(DATASETS_PATH, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])

#训练数据
class MVTec3DTrain(MVTec3D):
    def __init__(self, class_name, img_size):
        super().__init__(split="train", class_name=class_name, img_size=img_size)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def generate_train_rops(self,tiff_path,resized_organized_pc):
        organized_pc = resized_organized_pc
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))

        pcd_path = "/content/drive/MyDrive/pcdData/"+tiff_path[31:-5]+".pcd"
        if not (os.path.exists(pcd_path[:-8])):
            print(pcd_path[:-8])
            os.makedirs(pcd_path[:-8])
        if not (os.path.exists(pcd_path)): 
            print(pcd_path)
            o3d.io.write_point_cloud(pcd_path, o3d_pc)
        rops_path = "/content/drive/MyDrive/ropsData/"+tiff_path[31:-5]+".txt"
        if not(os.path.exists(rops_path[:-8])):
            print(rops_path[:-8])
            os.makedirs(rops_path[:-8])
        if not (os.path.exists(rops_path)):
            os.system("/content/3D-ADS/pcl_preprocess/build/rops_feature {} {}".format(pcd_path,rops_path))
        
        rops_feature = np.loadtxt(rops_path)
        return rops_feature




    #img_tot_paths样本路径列表[n,2]，[n][0]是rgb，[n][1]是tiff。tot_labels对应样本的标签[n]
    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_paths.sort()
        tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    #样本个数n
    def __len__(self):
        return len(self.img_paths)

    #遍历每个样本
    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        # #读RGB图像
        # img = Image.open(rgb_path).convert('RGB')
        # #降采样rgb
        # img = self.rgb_transform(img)
        img = np.zeros(0)
        #tiff转点云
        organized_pc = read_tiff_organized_pc(tiff_path)
        #降采样点云并转换为（z,x,y）
        resized_organized_pc = resize_organized_pc(organized_pc)

        # #点云生成3通道的深度图
        # depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        # #降采样深度图并转换为（3,x,y）
        # resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_depth_map_3channel = np.zeros(0)
        

        rops_feature = self.generate_train_rops(tiff_path,resized_organized_pc)

        return (img, resized_organized_pc, resized_depth_map_3channel,rops_feature), label


class MVTec3DTest(MVTec3D):
    def __init__(self, class_name, img_size):
        super().__init__(split="test", class_name=class_name, img_size=img_size)
        self.gt_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def generate_test_rops(self,tiff_path,resized_organized_pc):
        organized_pc = resized_organized_pc
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))

        pcd_path = "/content/drive/MyDrive/pcdData/"+tiff_path[31:-5]+".pcd"
        if not (os.path.exists(pcd_path[:-8])):
            print(pcd_path[:-8])
            os.makedirs(pcd_path[:-8])
        if not (os.path.exists(pcd_path)): 
            print(pcd_path)
            o3d.io.write_point_cloud(pcd_path, o3d_pc)
        rops_path = "/content/drive/MyDrive/ropsData/"+tiff_path[31:-5]+".txt"
        if not(os.path.exists(rops_path[:-8])):
            print(rops_path[:-8])
            os.makedirs(rops_path[:-8])
        if not (os.path.exists(rops_path)):
            os.system("/content/3D-ADS/pcl_preprocess/build/rops_feature {} {}".format(pcd_path,rops_path))
        
        rops_feature = np.loadtxt(rops_path)
        return rops_feature
             
        

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        #测试的类别
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        # img_original = Image.open(rgb_path).convert('RGB')
        # img = self.rgb_transform(img_original)
        img = np.zeros(0)

        organized_pc = read_tiff_organized_pc(tiff_path)
        resized_organized_pc = resize_organized_pc(organized_pc)

        # depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        # resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_depth_map_3channel = np.zeros(0)


        rops_feature = self.generate_test_rops(tiff_path, resized_organized_pc)

        if gt == 0:
            depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
            resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else:
            #转换为灰度图
            gt = Image.open(gt).convert('L')
            #降采样
            gt = self.gt_transform(gt)
            #二值化 大于0.5置1，小于置0
            gt = torch.where(gt > 0.5, 1., .0)

        

        return (img, resized_organized_pc, resized_depth_map_3channel,rops_feature), gt[:1], label


def get_data_loader(split, class_name, img_size):
    if split in ['train']:
        dataset = MVTec3DTrain(class_name=class_name, img_size=img_size)
    elif split in ['test']:
        dataset = MVTec3DTest(class_name=class_name, img_size=img_size)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader
