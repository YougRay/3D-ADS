import tifffile as tiff
import torch


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img


def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0)
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode='nearest')
    if tensor_out:
        #改变点云序列（z,x,y)并降采样到指定长宽
        #如果dim=0只有1个，压缩
        return torch_resized_organized_pc.squeeze(dim=0)
    else:
        #长宽采样到指定后，还原至（x,y,z）
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).numpy()


def organized_pc_to_depth_map(organized_pc):
    return organized_pc[:, :, 2]
