import os
import torch 
import torch.utils.data as data
import numpy as np
import pyvista as pv

'''
If scale factor is -1 it will calcuate the scale factor from the meshes
For no scaling set scale factor to 1
'''
class MeshDataset(data.Dataset):
    def __init__(self,
                 mesh_dir,
                 npoints=10000,
                 scale_factor=-1,
                 subsample=-1,
                 missing_percent=0,
                 set_type='test'):
        self.num_points = npoints
        self.mesh_dir = mesh_dir
        self.point_sets = []
        self.names = []
        self.missing_percent = missing_percent
        self.set_type = set_type
        
        calc_scale_factor = 0
        min_points = 1e8
        for file in sorted(os.listdir(mesh_dir)):
            points = np.array(pv.read(mesh_dir+file).points)
            if np.max(np.abs(points)) > calc_scale_factor:
                calc_scale_factor = np.max(np.abs(points))
            if points.shape[0] < min_points:
                min_points = points.shape[0]
            self.point_sets.append(points)
            self.names.append(file.replace(".vtk",""))
        self.min_points = min_points

        if scale_factor == -1:
            self.scale_factor = float(calc_scale_factor)
        else:
            self.scale_factor = scale_factor

        if subsample != -1:
            sorted_indices = np.load(mesh_dir + "../importance_sampling_indices.npy")
            indices = sorted_indices[:int(subsample)]
            pts, nms = [], []
            for index in indices:
                pts.append(self.point_sets[index])
                nms.append(self.names[index])
            self.point_sets = pts
            self.names = nms
            
    def get_scale_factor(self):
        return self.scale_factor

    def __getitem__(self, index):
        full_point_set = self.point_sets[index]
        name = self.names[index]
        full_point_set = full_point_set / self.scale_factor
        if self.missing_percent == 0:
            partial_point_set = full_point_set
        else:
            if self.set_type == 'train':
                seed = np.random.randint(len(full_point_set))
            else:
                seed = 0 # consistent testing
            distances = np.linalg.norm(full_point_set - full_point_set[seed], axis=1)
            sorted_points = full_point_set[np.argsort(distances)]
            partial_point_set = sorted_points[int(len(full_point_set)*self.missing_percent):]

        # partial
        if self.num_points > len(partial_point_set):
            replace = True
        else: 
            replace = False
        choice = np.random.choice(len(partial_point_set), self.num_points, replace=replace)
        partial = torch.FloatTensor(partial_point_set[choice, :])
        # ground truth 
        choice = np.random.choice(len(full_point_set), self.min_points, replace=False)
        gt = torch.FloatTensor(full_point_set[choice, :])
        
        return partial, gt, name

    def __len__(self):
        return len(self.point_sets)