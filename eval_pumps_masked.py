import colorsys
import os
import random
from colorsys import hsv_to_rgb

import PySide6
import numpy as np
import torch
from torch import nn

from models.MotionDecoder import MotionDecoder
from models.MotionTransformer import MotionTransformer
from models.PoseEncoder import PoseEncoder, PointCloudDecoder
from models.modules import PointTransformer
from vis.visualiser import Visualiser

if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    
    N_FRAMES = 128
    N_POINTS = 256
    STD = 0.05
    
    device = torch.device("cuda")
    
    data_path = sys.argv[1] # should be motion file path without the extension (".pt" or ".skel.pt")
    skeleton = torch.load(data_path + ".skel.pt")
    motion = torch.load(data_path + ".pt")
    start_idx = random.randint(0, motion.shape[0] - N_FRAMES - 1)
    motion = motion[start_idx:start_idx + N_FRAMES].unsqueeze(0)
    
    mask_type = sys.argv[2]
    if mask_type == "-c":
        mask = torch.zeros(N_FRAMES, dtype=torch.bool)
        keys = sys.argv[3].split(",")
        keys = [int(k) for k in keys]
        mask[keys] = 1
    else:
        mask = torch.ones(N_FRAMES, dtype=torch.bool)
        mask_val = int(sys.argv[3])
        if mask_type == "-i" or mask_type == "-iR":
            masked_idx = torch.randperm(N_FRAMES - 2)[:mask_val]
            mask[masked_idx] = 0
        elif mask_type == "-iU":
            mask = mask - 1
            for i in range(0, N_FRAMES, mask_val):
                mask[i] = 1
            mask[-1] = 1
        elif mask_type == "-t":
            start_idx = random.randint(1, N_FRAMES - mask_val - 1)
            mask[start_idx:start_idx + N_FRAMES] = 0
        elif mask_type == "-pS":
            mask[:mask_val] = 0
        elif mask_type == "-pE" or mask_type == "-p":
            mask[-mask_val:] = 0
        else:
            raise ValueError("Mask type not found! Please use one of the following: -i -iR -iU -t -p -pS -pE -c")
    print("Using key mask:")
    print(mask)
    mask = mask.unsqueeze(0).to(device)
    
    point_encoder = PoseEncoder(n_bgroups=5,
                                  d_model=32,
                                  k=8).to(device)
    point_encoder.load_state_dict(torch.load("./checkpoints/latent100000/encoder.pt"))
    point_encoder.eval()

    motion_model = MotionTransformer(d_input=512 + 3,
                                     d_model=512,
                                     d_output=512 + 3,
                                     n_heads=8,
                                     n_layers=8,
                                     max_length=N_FRAMES).to(device)
    motion_model.load_state_dict(torch.load("./checkpoints/motion50000/model.pt"))
    motion_model.eval()

    point_decoder = PointCloudDecoder(n_bgroups=5,
                                      d_embed=512,
                                      n_heads=8,
                                      n_layers=16,
                                      max_length=N_FRAMES,
                                      dropout=0.0).to(device)
    point_decoder.load_state_dict(torch.load("./checkpoints/latent100000/decoder.pt"))
    point_decoder.eval()

    camera_params = {'elevation': 0.0,
                     'azimuth': 180.0,
                     'distance': 75.0,
                     'center': PySide6.QtGui.QVector3D(0.0, 0.0, 0.0),
                     'fov': 10}
    vis = Visualiser(width=8, camera_params=camera_params)

    random.seed(600)
    torch.random.manual_seed(600)

    with torch.no_grad():
        p, q = skeleton.fk(motion[..., :3], motion[..., 3:], local_q=False)
        point_clouds, parent_joints = skeleton.generate_pointcloud(p, q, n_points=N_POINTS, std=STD, output_joints=True)
        bgroups = point_clouds[:, 0, :, 3:].clone()

        with torch.no_grad():
            embeds, offsets = point_encoder(torch.reshape(point_clouds, (batches * frames, n_points, -1)), return_offset=True)
            embeds = torch.reshape(embeds, (batches, frames, -1))
            offsets = torch.reshape(offsets, (batches, frames, 3))
        
        pred_embeds, pred_offset = motion_model(embeds, offsets, mask)
        pred_points = point_decoder(embeds, offsets, bgroups, apply_offset=True)

        # visualisation code below
        point_result = torch.stack([point_clouds[0, ..., :3], pred_points[0, ..., :3]], dim=0)
        point_result[0, ..., 2] += 1
        point_result[1, ..., 2] -= 1
        point_result = torch.reshape(point_result.transpose(0, 1), (N_FRAMES, -1, 3))

        heads = point_result.clone()
        tails = heads.clone()
        tails[..., 2] += 0.01

        colour_bgroups = torch.tensor([colorsys.hsv_to_rgb(i / 5, 1, 1) for i in range(5)], device=device)
        point_bgroups = skeleton.joint_bgroups[parent_joints]
        point_bgroups = point_bgroups.repeat(2, 1)
        point_colours = point_bgroups.unsqueeze(-1) * colour_bgroups
        colours = torch.sum(point_colours, dim=1)
        colours = torch.cat([colours, torch.ones(*colours.shape[:-1], 1, device=device)], dim=-1)
        colours = torch.broadcast_to(colours, (heads.shape[0], heads.shape[1], 4))

        vis.update_data(heads, tails, colours) # top = original, bottom = prediction