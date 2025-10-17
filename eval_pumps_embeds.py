import colorsys
import math
import os
import random

import PySide6
import torch
from models.PoseEncoder import PoseEncoder, PointCloudDecoder
from data.skeleton import Skeleton
from vis.visualiser import Visualiser


if __name__ == "__main__":
    N_FRAMES = 128
    N_POINTS = 256
    STD = 0.05

    device = torch.device("cuda")
    
    data_path = sys.argv[1] # should be motion file path without the extension (".pt" or ".skel.pt")
    skeleton = torch.load(data_path + ".skel.pt")
    motion = torch.load(data_path + ".pt")
    start_idx = random.randint(0, motion.shape[0] - N_FRAMES - 1)
    motion = motion[start_idx:start_idx + N_FRAMES].unsqueeze(0)

    embedding_model = PoseEncoder(n_bgroups=5,
                                  d_model=32,
                                  k=8).to(device)
    embedding_model.load_state_dict(torch.load("./checkpoints/latent100000/encoder.pt"))
    embedding_model.eval()

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
    vis = Visualiser(width=4, camera_params=camera_params)

    torch.manual_seed(600)
    random.seed(600)

    with torch.no_grad():
        p, q = skeleton.fk(motion[..., :3], motion[..., 3:], local_q=False)
        point_clouds, parent_joints = skeleton.generate_pointcloud(p, q, n_points=N_POINTS, std=STD,
                                                                   output_joints=True)

        offset = torch.mean(point_clouds[..., :3], dim=(1, 2), keepdim=True)
        point_clouds[..., :3] -= offset
        real_p -= offset

        bgroups = point_clouds[:, 0, :, 3:].clone()

        encoder_input = torch.reshape(point_clouds, (N_FRAMES, N_POINTS, -1))
        embeds, offsets = embedding_model(encoder_input, return_offset=True)
        embeds = torch.reshape(embeds, (1, N_FRAMES, -1))
        offsets = torch.reshape(offsets, (1, N_FRAMES, 3))

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