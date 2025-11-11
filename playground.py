import numpy as np
import torch
from vine_prune.utils.io import load_depth, save_depth
import cv2
import shutil

K = np.loadtxt('/home/hfreeman/harry_ws/repos/pruner_track/datasets/DEMOS/chili_place/demos/GX019688/cam_K.txt')
K = torch.from_numpy(K)
# K[0] /= 960
# K[1] /= 720
torch.save(K, 'example_harry/intrinsics.pt')

depth = load_depth('/home/hfreeman/harry_ws/repos/pruner_track/datasets/DEMOS/chili_place/demos/GX019688/depth/000000.png', scale=True)
# save_depth('example_harry/depth.png', depth, scale=True)
# depth = 10 * depth
save_depth('example_harry/depth.png', depth, scale=True)

im = cv2.imread('/home/hfreeman/harry_ws/repos/pruner_track/datasets/DEMOS/chili_place/demos/GX019688/undistorted/000000.jpg')
cv2.imwrite('example_harry/rgb.png', im)

shutil.copyfile('/home/hfreeman/harry_ws/repos/pruner_track/datasets/DEMOS/chili_place/demos/GX019688/masks/objects/chili_1/mask_obj/000000.png',
                'example_harry/mask.png')
