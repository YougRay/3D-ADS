from utils.mvtec3d_util import *
import open3d as o3d
import numpy as np
import torch
from feature_extractors.features import Features

def get_rops_features(organized_pc, histograms,voxel_size=0.05):
    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    print(histograms.shape)

    rops = histograms.data
    full_rops = np.zeros((unorganized_pc.shape[0], rops.shape[1]), dtype=rops.dtype)
    full_rops[nonzero_indices, :] = rops
    full_rops_reshaped = full_rops.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], rops.shape[1]))
    full_rops_tensor = torch.tensor(full_rops_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
    return full_rops_tensor

class ROPSFeatures(Features):
    def add_sample_to_mem_bank(self, sample):
        rops_feature_maps = get_rops_features(sample[1],sample[3])
        rops_feature_maps_resized = self.resize(self.average(rops_feature_maps))
        rops_patch = rops_feature_maps_resized.reshape(rops_feature_maps_resized.shape[1], -1).T
        self.patch_lib.append(rops_patch)

    def predict(self, sample, mask, label):
        rops_feature_maps = get_rops_features(sample[1],sample[3])
        rops_feature_maps_resized = self.resize(self.average(rops_feature_maps))
        patch = rops_feature_maps_resized.reshape(rops_feature_maps_resized.shape[1], -1).T

        self.compute_s_s_map(patch, rops_feature_maps_resized.shape[-2:], mask, label)