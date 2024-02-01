import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from nerf_model import *

from dataloader.load_llff import load_llff_data
#from load_deepvoxels import load_dv_data
#from load_blender import load_blender_data
#from load_LINEMOD import load_LINEMOD_data

# custom
from nerf_utils import *
from options import config_parser
from bokeh_utils import render_path_bokeh, render_bokeh


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'param'), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, bokeh_param = create_nerf(args, N_image=len(i_train))
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    #render_poses = torch.Tensor(render_poses).to(device)
    # render_poses = np.load(args.pose_path)
    # render_poses = torch.tensor(render_poses[:, :3, :4]).cuda()
    
    
    if args.expname == 'amusement22':
      if args.pose == 'pose1':
        print("amuse pose1")
        custompose = np.array([[[ 0.9995, -0.0126, -0.0306, -0.0496],
        [ 0.0119,  0.9997, -0.0228,  0.2541],
        [ 0.0308,  0.0225,  0.9993, -0.0434]]])
      elif args.pose == 'pose2':
        print("amuse pose2")
        custompose = np.array([[[ 1.0000e+00,  2.2572e-11,  3.6962e-10,  5.2023e-09],
         [ 2.7930e-14,  9.9814e-01, -6.1030e-02, -2.1917e-01],
         [-3.7031e-10,  6.1030e-02,  9.9814e-01, -1.5379e-01]]])
      else:
        raise RuntimeError("wrong pose!")
    
    elif args.expname == 'gink22':
      if args.pose == 'pose1':
        print("gink pose1")
        custompose = np.array([[[ 0.9967,  0.0343,  0.0738,  0.0618],
        [-0.0289,  0.9969, -0.0734, -0.3552],
        [-0.0761,  0.0710,  0.9946,  0.2617]]])
      elif args.pose == 'pose2':
        custompose = np.array([[[ 1.0000e+00, -1.0006e-09, -1.3122e-08, -5.5355e-09],
        [ 7.2009e-11,  9.9751e-01, -7.0577e-02, -2.6847e-01],
        [ 1.3160e-08,  7.0577e-02,  9.9751e-01, -1.2141e-01]]])
      else:
        raise RuntimeError("wrong pose!")
      
    elif args.expname == 'bear22':
      if args.pose == 'pose1':
        custompose = np.array([[[ 0.9997,  0.0035, -0.0224, -0.1230],
        [-0.0047,  0.9987, -0.0504,  0.3181],
        [ 0.0222,  0.0505,  0.9985,  0.1824]]])
      elif args.pose == 'pose2':
        custompose = np.array([[[ 1.0000e+00,  9.3144e-09, -3.3643e-09, -3.8276e-09],
        [-9.6896e-09,  9.9043e-01, -1.3800e-01, -5.1021e-01],
        [ 2.0467e-09,  1.3800e-01,  9.9043e-01, -1.9862e-01]]])
      else:
        raise RuntimeError("wrong pose!")
      
    elif args.expname == 'sheep22':
      if args.pose == 'pose1':
        custompose = np.array([[[ 0.9977, -0.0130, -0.0660, -0.7117],
        [ 0.0126,  0.9999, -0.0072,  0.3270],
        [ 0.0661,  0.0063,  0.9978, -1.2527]]])
      elif args.pose == 'pose2':
        custompose = np.array([[[ 1.0000e+00,  8.8290e-09, -1.4256e-08,  6.5774e-09],
        [-1.0785e-08,  9.8964e-01, -1.4360e-01, -5.2190e-01],
        [ 1.2840e-08,  1.4360e-01,  9.8964e-01, -6.5429e-01]]])
      else:
        raise RuntimeError("wrong pose!")
      
    elif args.expname == 'chrysan22':
      if args.pose == 'pose1':
        custompose = np.array([[[ 0.9998, -0.0184,  0.0100, -0.2852],
        [ 0.0189,  0.9985, -0.0517, -0.2063],
        [-0.0091,  0.0519,  0.9986,  0.3316]]])
      elif args.pose == 'pose2':
        custompose = np.array([[[ 1.0000e+00,  1.9624e-09, -3.3454e-09,  3.2047e-08],
        [-2.1601e-09,  9.9819e-01, -6.0185e-02, -2.3232e-01],
        [ 3.2212e-09,  6.0185e-02,  9.9819e-01, -4.3198e-01]]])
      else:
        raise RuntimeError("wrong pose!")
      
    elif args.expname == 'snowman22':
      if args.pose == 'pose1':
        custompose = np.array([[[ 0.9892,  0.0102,  0.1463,  0.2678],
        [ 0.0023,  0.9963, -0.0856, -0.2579],
        [-0.1466,  0.0850,  0.9855,  0.1523]]])
      elif args.pose == 'pose2':
        custompose = np.array([[[ 1.0000e+00, -4.2176e-10,  9.1039e-10, -1.8182e-08],
        [ 5.9473e-10,  9.7994e-01, -1.9929e-01, -7.4497e-01],
        [-8.0807e-10,  1.9929e-01,  9.7994e-01, -3.5341e-01]]])
      else:
        raise RuntimeError("wrong pose!")
    
    elif args.expname == 'applemint22':
      if args.pose == 'pose1':
        custompose = np.array([[[ 1.0000e+00,  3.8563e-10,  1.5768e-10,  9.1873e-09],
         [-3.7260e-10,  9.9714e-01, -7.5626e-02, -2.6982e-01],
         [-1.8639e-10,  7.5626e-02,  9.9714e-01, -1.3646e-01]]])
      else:
        raise RuntimeError("wrong pose!")
    
    elif args.expname == 'xmas22':
      if args.pose == 'pose1':
        custompose = np.array([[[ 1.0000e+00, -1.9322e-09,  5.4578e-10, -2.9506e-09],
         [ 1.9826e-09,  9.9323e-01, -1.1618e-01, -4.5130e-01],
         [-3.1760e-10,  1.1618e-01,  9.9323e-01, -1.2837e-01]]])
      else:
        raise RuntimeError("wrong pose!")
    
  
    elif args.expname == 'boyandgirl22':
      if args.pose == 'pose1':
        custompose = np.array([[[ 1.0000e+00, -5.6444e-09,  1.3424e-09, -1.5922e-09],
         [ 5.7099e-09,  9.9847e-01, -5.5263e-02, -1.9367e-01],
         [-1.0284e-09,  5.5263e-02,  9.9847e-01, -1.2536e-01]]])
      else:
        raise RuntimeError("wrong pose!")
    
    else:
      raise RuntimeError("wrong scene!")
    
    render_poses = torch.tensor(custompose).cuda()
    
    # F number
    asize = np.array([ 2.        ,  2.17240147,  2.35966408,  2.56306886,  2.78400728,
    3.02399075,  3.28466098,  3.56780117,  3.87534826,  4.20940613,
    4.57226003,  4.96639221,  5.39449887,  5.85950865,  6.3646026 ,
    6.91323603,  7.50916206,  8.15645735,  8.85954998,  9.6232497 ,
    10.45278091, 11.35381831, 12.3325258 , 13.3955986 , 14.55030905,
    15.80455639, 17.16692078, 18.64672198, 20.25408313, 22.,
1233.22537816, 1019.42765155,  842.69490001,  696.6013659 ,
    575.835291  ,  476.00578838,  393.48319582,  325.26710636,
    268.87727761,  222.26345364,  183.73082048,  151.87838505,
    125.54803699,  103.78244137,   85.79023133,   70.91723508,
    58.62269111,   48.45958686,   40.05840595,   33.1136931 ,
    27.3729482 ,   22.62744572,   18.70464576,   15.46192078,
    12.78136979,   10.56553167,    8.73384162,    7.21970194,
        5.9680606 ,    4.93340967 ])
    asize = asize - 2
    asize = np.clip(asize, a_min=0, a_max=22)
    asize = 1 - asize / 22
    asize = asize * 22
    arr = asize        
    
    rgbs_list = []
    n_frame = 60
    
    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            for i in tqdm(range(n_frame)):

                rgbs, _ = render_path_bokeh(render_poses, hwf, K, args.chunk, render_kwargs_test,
                                    K_bokeh=arr[i], gamma=4, disp_focus=30/255, defocus_scale=1,
                                    gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
                rgbs_list.append(rgbs[0])

            #rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, f'video_{args.pose}.mp4'), to8b(np.array(rgbs_list)), fps=15, quality=8)

            return


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
