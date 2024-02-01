import subprocess
import os

# exps = ["configs/custom/amusementpark/tx_amuse4_full.txt",
#         'configs/custom/applemint/tx_applemint4_full.txt',
#         'configs/custom/bear/tx_bear4_full.txt',
#         'configs/custom/gink/tx_gink4_full.txt',
#         'configs/custom/snowman/tx_snowman4_full.txt'
#         ]

exps = [
        'applemint/tx_applemint22',
        'boyandgirl/tx_boyandgirl22',
        'xmas/tx_xmas22',
]

# pose = ['AmusementPark_video_render_poses.npy',
#         'AppleMint_video_render_poses.npy',
#         'Bear_video_render_poses.npy',
#         'Gink_video_render_poses.npy',
#         'Snowman2_video_render_poses.npy',
#         ]

pose = [
        'pose1',
        'pose1',
        'pose1',

]

for scene, ginname in zip(exps, pose):
    subprocess.run(f"CUDA_VISIBLE_DEVICES=7 python defocus_video.py --config configs/custom/{scene}.txt  --pose {ginname} --render_only", shell=True)
