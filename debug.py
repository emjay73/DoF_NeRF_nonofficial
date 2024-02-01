import os, shutil
from tqdm import tqdm

tgt_path = "/home/nas2_userF/gyojunggu/gyojung/lensnerf/DoF-NeRF/output"
for scene, scene_upper in [("amusement", "AmusementPark"), ("applemint", "AppleMint"), ("bear", "Bear"), ("boyandgirl", "BoyAndGirl"),
                         ("chrysan", "Chrysanthemum"), ("gink", "Gink"), ("sheep", "Sheep"), ("snowman", "Snowman2"), ("xmas", "Xmas")]:
    for F, F_upper in tqdm([("4", "F4"), ("5dot6", "F5dot6"), ("8", "F8"), ("_mix", "Fmix")]):
        os.makedirs(F"results/deblur/{scene_upper}/train_{F_upper}_test_F22", exist_ok=True)
        files = os.listdir(os.path.join(tgt_path, scene+F, "testset_400000"))
        for f in files:
            shutil.copy2(os.path.join(tgt_path, scene+F, "testset_400000", f), F"results/deblur/{scene_upper}/train_{F_upper}_test_F22")