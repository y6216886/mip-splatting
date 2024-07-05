#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys
import torch
import pathlib
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, List, BinaryIO
import os

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
def save_images(img, img_fname, txt_label=None):
    if not os.path.isdir(os.path.dirname(img_fname)):
        os.makedirs(os.path.dirname(img_fname), exist_ok=True)
    im = Image.fromarray(img)
    if txt_label is not None:
        draw = ImageDraw.Draw(im)
        txt_font = ImageFont.load_default()
        draw.text((10, 10), txt_label, fill=(0, 0, 0), font=txt_font)
    im.save(img_fname)