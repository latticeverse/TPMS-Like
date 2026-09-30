import argparse
import datetime
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import models, _global
from tpmslike.boundary import read_boundary

try:
    import mcubes
except ImportError:
    mcubes = None
from skimage.measure import marching_cubes as skimage_marching_cubes

#wirte mesh
def write_ply_triangle(name, vertices, triangles):
    """Write a triangular mesh using the legacy Deep Currents axis order.

    Marching-cubes indices are returned in ``(y, x, z)`` array order for the
    grid construction used by the original implementation.  Swapping the
    first two coordinates here is therefore part of the file format, rather
    than a display-only choice.
    """
    os.makedirs(os.path.dirname(name) or ".", exist_ok=True)
    fout = open(name, "w", encoding="utf-8")
    for ii in range(len(vertices)):
        vertex = vertices[ii]
        fout.write(f"v {float(vertex[1]):.9g} {float(vertex[0]):.9g} {float(vertex[2]):.9g}\n")
    for ii in range(len(triangles)):
        fout.write("f " +str(triangles[ii,0] + 1)+" "+str(triangles[ii,1] + 1)+" "+str(triangles[ii,2] + 1)+"\n")
    fout.close()

#read boundary
def init(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    boundary = read_boundary(args.boundary)
    bdry_num = boundary.n_curves
    bdry_verts = torch.from_numpy(boundary.points).to(_global.device)
    print(bdry_num,bdry_verts.shape)
    return bdry_num, bdry_verts


def marching_cubes(volume, level):
    """Use PyMCubes when available and fall back to scikit-image."""
    if mcubes is not None:
        return mcubes.marching_cubes(volume, level)
    result = skimage_marching_cubes(volume, level=level)
    return result[0], result[1]



#适用于三次镜像，找每个面上的边界点
def group_bdry_verts_rrr(bdry_verts_single):
    eps = 1e-6
    ids = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            face_ids = torch.nonzero(
                torch.abs(bdry_verts_single[:, axis] - sign * _global.bdry_d) < eps,
                as_tuple=False,
            ).flatten()
            if face_ids.numel() == 0:
                raise ValueError(f"Boundary has no samples on face axis={axis}, sign={sign:g}")
            ids.append(face_ids)
    print("boundary samples per face:", [int(face.numel()) for face in ids])
    return ids
    

#适用于三次平移，找对称点
def group_bdry_verts_ttt(bdry_verts_single):
    eps = 1e-6
    pairs = []
    for axis in range(3):
        other_axes = [item for item in range(3) if item != axis]
        plus = torch.nonzero(
            torch.abs(bdry_verts_single[:, axis] - _global.bdry_d) < eps,
            as_tuple=False,
        ).flatten()
        minus = torch.nonzero(
            torch.abs(bdry_verts_single[:, axis] + _global.bdry_d) < eps,
            as_tuple=False,
        ).flatten()
        if plus.numel() != minus.numel() or plus.numel() == 0:
            raise ValueError(f"Opposite faces on axis {axis} have different sample counts")
        minus_coords = bdry_verts_single[minus][:, other_axes]
        matched = []
        for point_id in plus:
            coord = bdry_verts_single[point_id, other_axes]
            distance = torch.max(torch.abs(minus_coords - coord), dim=1).values
            nearest = torch.argmin(distance)
            if distance[nearest] >= eps:
                raise ValueError(f"No translated counterpart found for boundary point {int(point_id)}")
            matched.append(minus[nearest])
        pairs.extend((plus, torch.stack(matched)))
    print("opposite face sample pairs:", [int(face.numel()) for face in pairs])
    return pairs

def main(args, index):
    if args.type== 'rrr':
        _global.reflect_type=1
    elif args.type== 'ttt':
        _global.reflect_type=2
    else:
        _global.reflect_type=3
    _t = init(args)
    bdry_verts_single = _t[1].float()
    bdry_num = (int)(_t[0])


    face_contain_point_id=1
    if args.type=='rrr':
        face_contain_point_id = group_bdry_verts_rrr(bdry_verts_single)# t translation r reflection
    else:
        face_contain_point_id = group_bdry_verts_ttt(bdry_verts_single)# t translation r reflection
    

    print(bdry_verts_single.size())
    bdry_mul = bdry_verts_single.contiguous().view((bdry_num, (int)(bdry_verts_single.size()[0] / bdry_num) , 3))
    print('boundary size:',bdry_mul.size())
    model = models.SurfaceModel(bdry=bdry_mul, rff_sigma=args.rff_sigma)

    model.to(_global.device)
    model.train()

    optimizer = optim.Adam( [{'params': model.parameters()}], lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    writer = SummaryWriter(os.path.join(
        args.out, datetime.datetime.now().strftime('train-%m%d%y-%H%M%S')),
        flush_secs=1)

    it = 0
    pbar = tqdm(total=args.n_iterations)
    for it in range(args.n_iterations): 

        optimizer.zero_grad()


        x = torch.empty(args.n_samples, 3).uniform_(-1, 1).to(_global.device)#采样点
        x = torch.cat([bdry_verts_single, x]).to(_global.device)#采样点+输入边界点
        x.requires_grad = True

        out = model(x, total_bdry_point_num=bdry_verts_single.size()[0], face_contain_point_id=face_contain_point_id)
        loss = (out['current'].norm(p=2, dim=-1)).mean() + out['c1_continuity'].mean()

	#+ out['c1_continuity'].mean() 
      
   
      
        loss.backward()
        
        optimizer.step()
        if it % 10000 == 0:
            scheduler.step()

        pbar.set_postfix({
            'loss': loss.item(),
        })
        pbar.update(1)
        writer.add_scalar('loss', loss.item(), it)
        
        it += 1
    #marching cubes 提取网格
    mean_total = []
    min = 100
    max = -100
    for i in range(bdry_num):

        bdry_verts_fx = model(bdry_mul[i], only_f = True)
        bdry_verts_fx_min = bdry_verts_fx.min()
        bdry_verts_fx_max = bdry_verts_fx.max()
        mean_total.append(bdry_verts_fx.mean())
        # mean = bdry_verts_fx.mean()
        _min = bdry_verts_fx_min.detach().cpu().numpy()
        _max = bdry_verts_fx_max.detach().cpu().numpy()
        if min > _min:
            min = _min
        if max < _max:
            max = _max

        print("min", bdry_verts_fx_min)
        print("max", bdry_verts_fx_max)
        print("mean", bdry_verts_fx.mean())


    grid_size = args.grid_size
    d = _global.bdry_d #提取立方体大小 分辨率=_global.bdry_d*2 / grid_size
    x = np.arange(-d,d + grid_size, grid_size)
    y = np.arange(-d,d + grid_size, grid_size)
    z = np.arange(-d,d + grid_size, grid_size)
    
    resolution = (int)( (d * 2) / grid_size) + 1
    sample_x = np.array(np.meshgrid(x,y,z))
    sample_x = sample_x .reshape((3,resolution * resolution * resolution))
    sample_x = sample_x .transpose()
    sample_x = torch.from_numpy(sample_x).to(_global.device).to(torch.float32)
    
    grid_fx = np.empty([sample_x.size()[0],1], dtype = np.float32) 
    for i in range(resolution):
        with torch.no_grad():
            t = model(sample_x[resolution * resolution * i:resolution * resolution * (i+1), :], only_f = True)
            t = t.detach().cpu().numpy()
        grid_fx[resolution * resolution * i: resolution * resolution * (i + 1), ...] = t
    
    mean_total_average = 0
    max_value=0
    min_value=0
    grid_fx = grid_fx.reshape((resolution, resolution, resolution))
    # print('grid_fx shape',grid_fx.shape)
    # print(grid_fx[0][0][0])
    # print(grid_fx(0,0,0))
    
    _max=-1000.0
    _min=1000.0
    for i in range (0,resolution):
        for j in range (0,resolution):
            for k in range (0,resolution):
                if(grid_fx[i][j][k]>_max):
                    _max=grid_fx[i][j][k]
                if(grid_fx[i][j][k]<_min):
                    _min=grid_fx[i][j][k]
    step=(_max-_min)/10.0
    print(_max,_min,step,type(_max),type(_min),type(step))
    for i in np.arange (_min,_max,step):
        _vertices, _triangles = marching_cubes(grid_fx, i)
        _vertices = _vertices / ((resolution - 1) / (2.0 * d)) - d
        write_ply_triangle(args.out_debug + '/' + str(index)  + "_" + str(i)+ '_or_ms.obj', _vertices, _triangles) 
    for i in range(bdry_num):
        mean = mean_total[i].detach().cpu().numpy()
        mean_total_average += mean
        grid_fx = grid_fx.reshape((resolution, resolution, resolution))
        _vertices, _triangles = marching_cubes(grid_fx, mean)
        _vertices = _vertices / ((resolution - 1) / (2.0 * d)) - d
        write_ply_triangle(args.out + '/' + str(index)  + "_" + str(i)+ '_or_ms.obj', _vertices, _triangles) 
    
    mean_total_average /= bdry_num
    grid_fx = grid_fx.reshape((resolution, resolution, resolution))
    _vertices, _triangles = marching_cubes(grid_fx, mean_total_average)
    _vertices = _vertices / ((resolution - 1) / (2.0 * d)) - d
    # _vertices /= 2
    Type=args.type
    # if Type == 'ttt_a':
    #     Type='ttt'
    write_ply_triangle(args.out + '/' + str(index)  + "_" +Type+ '_or_ms_average.obj', _vertices, _triangles) 
    


    # torch.cuda.empty_cache()

if __name__ == '__main__':
    


    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='runs/deepcurrents')
    parser.add_argument('--out_debug', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=2**12)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_iterations', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--boundary', type=str, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--type', choices=('rrr', 'ttt', 'ttt_a'), default='rrr')
    parser.add_argument('--rff_sigma', type=float, default=2)
    parser.add_argument('--grid-size', type=float, default=0.04)
    parser.add_argument('--device', '--CUDAID', dest='device', default=None)
    
    
    args = parser.parse_args()
    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    _global.device = args.device
    if args.out_debug is None:
        args.out_debug = os.path.join(args.out, 'debug')
    print(args)
    index=args.index
    main(args, index)
        # torch.cuda.empty_cache()
