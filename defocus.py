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
    render_poses = torch.Tensor(render_poses).to(device)

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

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path_bokeh(render_poses, hwf, K, args.chunk, render_kwargs_test,
                                  K_bokeh=1, gamma=4, disp_focus=30/255, defocus_scale=1,
                                  gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            #rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb_nhw = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        train_idx = np.arange(rays_rgb_nhw.shape[0])
        W_anchors = np.arange(79)
        H_anchors = np.arange(52)
        # W_anchor = 0~78, H_anchor = 0~51
        rays_rgb = np.reshape(rays_rgb_nhw, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        rays_rgb_nhw = rays_rgb_nhw.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)
        rays_rgb_nhw = torch.Tensor(rays_rgb_nhw).to(device)


    N_iters = args.N_iters + 1
    bokeh_iters = args.bokeh_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    for i in tqdm(trange(start, N_iters)):
        time0 = time.time()
        if i < bokeh_iters:
            # Sample random ray batch
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
                
            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)

        else:
            if i == bokeh_iters:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lrate_bokeh
                
            # Random from one image
            img_i = np.random.choice(train_idx)
            batch = rays_rgb_nhw[img_i]  # [H, W, 2+1, 3]
            H_anchor = np.random.choice(H_anchors)
            W_anchor = np.random.choice(W_anchors)
            batch_patch = batch[(16*H_anchor):(16*H_anchor+48), (16*W_anchor):(16*W_anchor+48), :, :]  # [48, 48, 2+1, 3]
            batch = torch.reshape(batch_patch, [-1, 3, 3])  # [48*48, 2+1, 3]
            #batch_idx = np.random.choice(batch.shape[0], size=[N_rand], replace=False)  # get 1024 rays from the patch
            #batch = batch_flatten[batch_idx]  # [1024, 2+1, 3]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
            
            # target = images[img_i]
            # target = torch.Tensor(target).to(device)
            # pose = poses[img_i, :3,:4]

            # if N_rand is not None:
            #     rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            #     if i < args.precrop_iters:
            #         dH = int(H//2 * args.precrop_frac)
            #         dW = int(W//2 * args.precrop_frac)
            #         coords = torch.stack(
            #             torch.meshgrid(
            #                 torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
            #                 torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
            #             ), -1)
            #         if i == start:
            #             print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
            #     else:
            #         coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            #     coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            #     select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            #     select_coords = coords[select_inds].long()  # (N_rand, 2)
            #     rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            #     rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            #     batch_rays = torch.stack([rays_o, rays_d], 0)
            #     target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True, **render_kwargs_train)
            para_K_bokeh, para_disp_focus = bokeh_param()

            rgbs = torch.reshape(rgb, [48, 48, 3])
            disps = torch.reshape(disp, [48, 48])
            bokeh_classical = render_bokeh(rgbs=rgbs, disps=disps, K_bokeh=para_K_bokeh[img_i], gamma=4, disp_focus=para_disp_focus[img_i], defocus_scale=1)
            rgb = torch.reshape(bokeh_classical, [-1, 3])
            target_s = torch.reshape(target_s, [-1, 3])
        
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
            
            param_path = os.path.join(basedir, expname, 'param', '{:06d}.tar'.format(i))
            torch.save(bokeh_param.state_dict(), param_path)
            
        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            if 'mix' in args.expname:
                target_F_list = [("4", "F4")]
                fix = 'mix'
            else:
                target_F_list = [("4", "F4"), ("5dot6", "F5dot6"), ("8", "F8")]
                fix = 'F22'
                
            for F, F_upper in tqdm(target_F_list):
                print(f"{F_upper} starts")
                if 'mix' in args.expname:
                    target_F = args.expname.replace("_mix", F)
                else:
                    target_F = args.expname.replace("22", F)
                
                # load target F bokeh param
                bokeh_param = nerf_model.Bokeh_Param(len(i_train))
                
                if os.path.exists(os.path.join(basedir, target_F, 'param')):
                    param_ckpts = [os.path.join(basedir, target_F, 'param', f) for f in sorted(os.listdir(os.path.join(basedir, target_F, 'param'))) if 'tar' in f]
                    if len(param_ckpts) > 0:
                        ckpt_path = param_ckpts[-1]
                        print('Reloading bokeh parameters from ', ckpt_path)
                        bokeh_param.load_state_dict(torch.load(ckpt_path))
                    else:
                        print('No reloading bokeh parameters')
                else:
                    print('No reloading bokeh parameters')
                    
                para_K_bokeh, para_disp_focus = bokeh_param()
                target_K = torch.mean(para_K_bokeh).item()
                target_disp_f = torch.mean(para_disp_focus).item()
                
                testsavedir = os.path.join('results', "defocus", f"{os.path.basename(args.datadir).split('_')[0]}", f"train_{fix}_test_{F_upper}")
            
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', poses[i_test].shape)
                with torch.no_grad():
                    render_path_bokeh(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                            K_bokeh=target_K, gamma=4, disp_focus=target_disp_f, defocus_scale=1,
                            gt_imgs=images[i_test], savedir=testsavedir)
                                    
                print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
