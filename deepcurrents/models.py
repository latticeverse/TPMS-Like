
from locale import currency
from zlib import DEF_BUF_SIZE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
from . import _global
import cmath
import copy

class SurfaceModel(nn.Module):

    def __init__(self, bdry=None, rff_sigma=2, dim_z=0, n_data=None,
                 dim_extra_z=0):
        super().__init__()

        self.rff = utils.InputMapping(3, 2048, sigma=rff_sigma)
        self.f = nn.Sequential(
            nn.Linear(2048+dim_z+dim_extra_z, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 1)
        )        

        if dim_z > 0 and n_data is not None:
            self.embedding = nn.Embedding(n_data, dim_z)
            nn.init.normal_(self.embedding.weight, std=0.1)

        if bdry is not None:
            self.compute_alpha = lambda x: utils.biot_savart_3d(x, bdry[None])
        else:
            self.compute_alpha = utils.biot_savart_3d
        

    def forward(self, x, bdry=None, z=None, only_f=False, total_bdry_point_num=0, face_contain_point_id=None, is_penalty=False, bdry_verts_single = None):
        
        y = self.rff(x)
        f = self.f(y)
        if only_f:
            return f

        df = torch.autograd.grad(outputs=f,
                                 inputs=x,
                                 grad_outputs=torch.ones_like(f),
                                 create_graph=True,
                                 only_inputs=True)[0]

        alpha = self.compute_alpha(x[None,total_bdry_point_num:,:])
        # alpha = self.compute_alpha(x[None,total_bdry_point_num:,:], x[None,None,0:total_bdry_point_num,:])
        current = df[total_bdry_point_num:,:] + alpha.flatten(0, 1)
        
        c1_continuity = 1

        if _global.reflect_type==1:
            # 三次镜像约束
            # Keep the constraint normals on the same device as ``df``.  The
            # training path is normally CUDA, while ``torch.tensor`` defaults
            # to CPU and otherwise causes a device mismatch during the first
            # forward pass.
            device = df.device
            n1 = torch.tensor([1, 0., 0.], dtype=torch.float32, device=device)
            n2 = torch.tensor([0., 1, 0.], dtype=torch.float32, device=device)
            n3 = torch.tensor([0., 0., 1], dtype=torch.float32, device=device)
            
            normals = [n1, n2, n3]
            penalties = []
            for axis, normal in enumerate(normals):
                for face_id in (2 * axis, 2 * axis + 1):
                    ids = face_contain_point_id[face_id].to(device)
                    penalties.append(torch.abs(torch.sum(df[ids] * normal, dim=1)))
            c1_continuity = torch.cat(penalties, dim=0)
        
        if _global.reflect_type==2:
        #平移但对应点法向向量相反
            c1_continuity = torch.cat([torch.abs(-1.0 * df[face_contain_point_id[0]] - df[face_contain_point_id[1]]), 
                            torch.abs(-1.0 * df[face_contain_point_id[2]] - df[face_contain_point_id[3]]),  
                            torch.abs(-1.0 * df[face_contain_point_id[4]] - df[face_contain_point_id[5]])], dim=0)
        if _global.reflect_type==3:
        # 平移但对应点法向向量相同
            c1_continuity = torch.cat([torch.abs(df[face_contain_point_id[0]] - df[face_contain_point_id[1]]), 
                            torch.abs(df[face_contain_point_id[2]] - df[face_contain_point_id[3]]),  
                            torch.abs(df[face_contain_point_id[4]] - df[face_contain_point_id[5]])], dim=0)
        

        return {
            'f': f,
            'current': current,
            'df': df,
            'alpha': alpha,
            'c1_continuity': c1_continuity
        }

class BoundaryEncoder(nn.Module):

    def __init__(self, dim_z=256, n_boundaries=1):
        super().__init__()

        self.encoders = nn.ModuleList(nn.Sequential(
            nn.Conv1d(3, dim_z, kernel_size=5, padding=2,
                      padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(dim_z, dim_z, kernel_size=3, padding=1,
                      padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(dim_z, dim_z, kernel_size=3, padding=1,
                      padding_mode='circular'),
        ) for _ in range(n_boundaries))

    def forward(self, bdries):
        zs = []
        for encode, bdry in zip(self.encoders, bdries):
            zs.append(encode(bdry.permute(0, 2, 1).contiguous()).mean(-1))
        z = torch.cat(zs, dim=-1)
        return z
