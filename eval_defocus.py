import os

from metrics import compute_img_metric
from dists_pt import dists

from tqdm import tqdm

import torch
import torchvision
from PIL import Image

import shutil

totensor = torchvision.transforms.transforms.ToTensor()

scene = "AmusementPark"
src_path = "/home/nas4_user/minjungkim/DB_custom/NeRF/anerf_dataset/real_defocus_dataset/"
tgt_path = "/home/nas2_userF/gyojunggu/gyojung/lensnerf/DoF-NeRF/results/defocus"
F = "_F4"

for scene, scene_upper in [("amusement", "AmusementPark"), ("applemint", "AppleMint"), ("bear", "Bear"), ("boyandgirl", "BoyAndGirl"),
                         ("chrysan", "Chrysanthemum"), ("gink", "Gink"), ("sheep", "Sheep"), ("snowman", "Snowman2"), ("xmas", "Xmas")]:
    for F, F_upper in tqdm([("4", "F4"), ("5dot6", "F5dot6"), ("8", "F8")]):
        with torch.no_grad():
            print(scene+F)
            src_image_path = os.path.join(src_path, scene_upper+f"_{F_upper}", 'images_4')
            tgt_image_path = os.path.join(tgt_path, scene_upper, f"train_F22_test_{F_upper}")

            image_names = [x for x in os.listdir(tgt_image_path) if x.endswith(".png")]
            image_index = [int(x.split(".")[0]) for x in image_names]
            print(image_index)
            src_images = [totensor(Image.open(os.path.join(src_image_path, f"IMG_{int(idx)*8+1:04d}.JPG"))) for idx in image_index]
            src_image_batch = torch.stack(src_images)

            tgt_image_batch = torch.stack([totensor(Image.open(os.path.join(tgt_image_path, name))) for name in image_names])

            psnr, psnr_list = compute_img_metric(src_image_batch, tgt_image_batch, 'psnr')
            ssim, ssim_list = compute_img_metric(src_image_batch, tgt_image_batch, 'ssim')
            lpips, lpips_list = compute_img_metric(src_image_batch, tgt_image_batch, 'lpips')
            lpips = float(lpips)
            dists_list = dists(src_image_batch.cuda(), tgt_image_batch.cuda()).cpu().numpy()
            test_dists = sum(dists_list) / len(dists_list)


            with open("results/defocus_results.csv", 'a') as f:
                f.write(f"{scene_upper},F22,{F_upper}")
                for i in range(7):
                    try:
                        f.write(f",{i},{psnr_list[i].item():.8f},{ssim_list[i].item():.8f},{lpips_list[i].item():.8f},{dists_list[i]:.8f}")
                    except IndexError:
                        f.write(",,,,,")
                f.write(f"\n")


            if F == '4':
                tgt_image_path = os.path.join(tgt_path, scene_upper, f"train_mix_test_{F_upper}")

                image_names = [x for x in os.listdir(tgt_image_path) if x.endswith(".png")]
                image_index = [int(x.split(".")[0]) for x in image_names]

                src_images = [totensor(Image.open(os.path.join(src_image_path, f"IMG_{int(idx)*8+1:04d}.JPG"))) for idx in image_index]
                src_image_batch = torch.stack(src_images)

                tgt_image_batch = torch.stack([totensor(Image.open(os.path.join(tgt_image_path, name))) for name in image_names])

                psnr, psnr_list = compute_img_metric(src_image_batch, tgt_image_batch, 'psnr')
                ssim, ssim_list = compute_img_metric(src_image_batch, tgt_image_batch, 'ssim')
                lpips, lpips_list = compute_img_metric(src_image_batch, tgt_image_batch, 'lpips')
                lpips = float(lpips)
                dists_list = dists(src_image_batch.cuda(), tgt_image_batch.cuda()).cpu().numpy()
                test_dists = sum(dists_list) / len(dists_list)


                with open("results/defocus_results.csv", 'a') as f:
                    f.write(f"{scene_upper},Fmix,{F_upper}")
                    for i in range(7):
                        try:
                            f.write(f",{i},{psnr_list[i].item():.8f},{ssim_list[i].item():.8f},{lpips_list[i].item():.8f},{dists_list[i]:.8f}")
                        except IndexError:
                            f.write(",,,,,")
                    f.write(f"\n")
