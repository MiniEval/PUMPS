import os

from pytorch3d.loss import chamfer_distance
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data.skeleton import Skeleton
from models.MotionTransformer import SelfAttentionLayer
from models.modules import PointTransformer, RMSNorm
from utils.knn_loss import KNNLoss


class PoseEncoder(nn.Module):
    def __init__(self,
                 n_bgroups,
                 d_model=32,
                 k=16):
        super(PoseEncoder, self).__init__()

        d_latent = d_model * 16

        self.n_bgroups = n_bgroups

        self.encoder = PointTransformer(self.n_bgroups, d_model, k=k, dropout=0)
        self.norm = nn.LayerNorm(d_latent, elementwise_affine=False, bias=False)

    def forward(self, points, return_offset=False):
        # points: [batch, points, 3+bgroups]

        # normalise position
        offset = torch.mean(points[..., :3], dim=1, keepdim=True)
        p = points.clone()
        p[..., :3] -= offset

        x = self.encoder(p)
        x = self.norm(x)

        if return_offset:
            return x, offset.clone().detach()
        else:
            return x

    def evaluate_point_cloud(self, poses, skeleton, std=None, n_points=256, min_std=0.01, max_std=0.1):
        # poses: [batch, (frames optional), pose]
        # each pose contains 3 (root pos) + 4j (joint quat)
        # all quats global
        batches = poses.shape[0]
        if len(poses.shape) < 3:
            poses = poses.unsqueeze(1)
        frames = poses.shape[1]
        root_p = poses[..., :3]
        glob_q = torch.reshape(poses[..., 3:], (batches, frames, -1, 4))
        glob_p, _ = skeleton.fk(root_p, glob_q, local_q=False)

        z_aug = 1 - torch.rand(batches)
        z_aug = torch.reshape(z_aug, (batches, 1))
        z_aug = torch.cat([z_aug, torch.zeros(batches, 2), torch.sqrt(1 - (z_aug ** 2))], dim=-1)
        z_aug = torch.reshape(z_aug, (batches, 1, 1, 4)).to(poses.device)
        z_aug = torch.broadcast_to(z_aug, (batches, poses.shape[1], skeleton.n_joints, 4))

        glob_p = Skeleton._qrot(glob_p, z_aug)
        glob_q = Skeleton._qmul(z_aug, glob_q)

        if std is None:
            std = torch.rand((batches, 256, 1), device=poses.device) * (max_std - min_std) + min_std
        return skeleton.generate_pointcloud(glob_p, glob_q, n_points=n_points, std=std)


class PointCloudDecoder(nn.Module):
    def __init__(self,
                 n_bgroups,
                 d_embed,
                 n_heads=8,
                 n_layers=8,
                 max_length=256,
                 d_seed=128,
                 dropout=0.0):
        super(PointCloudDecoder, self).__init__()

        assert d_embed % n_layers == 0

        self.d_embed = d_embed
        self.d_seed = d_seed
        self.n_heads = n_heads
        self.n_bgroups = n_bgroups

        self.dropout = dropout

        self.embed_in = nn.Linear(d_embed + 3, d_embed)

        self.embed_attn = nn.ModuleList([SelfAttentionLayer(d_embed, n_heads, rotary=True, d_ff=d_embed * 4,
                                                            dropout=0, activation=nn.GELU(), norm=RMSNorm(d_embed))
                                         for _ in range(n_layers)])

        self.relu = nn.GELU()
        self.bgroup_proj = nn.Linear(d_embed, (d_embed // 4) * n_bgroups)
        self.point_proj = nn.Linear(self.d_seed, d_embed // 4)

        self.output_mlp1 = nn.Linear(d_embed // 4, d_embed // 4)
        self.output_mlp2 = nn.Linear(d_embed // 4, d_embed // 4)
        self.output_mlp3 = nn.Linear(d_embed // 4, d_embed // 4)
        self.output_proj = nn.Linear(d_embed // 4, 3)

    def batch_dropout(self, x):
        batches = x.shape[0]
        size = x.shape[-1]

        mask = []
        for b in range(batches):
            mask_ = np.ones(size, dtype=np.bool_)
            mask_[:round(size * self.dropout)] = 0
            np.random.shuffle(mask_)
            mask_ = torch.from_numpy(mask_).bool().to(x.device)
            mask.append(mask_)
        mask = torch.stack(mask, dim=0).unsqueeze(1)

        out = x * mask * (1.0 / (1.0 - self.dropout))
        return out

    def forward(self, embeds, offsets, bgroups, apply_offset=True):
        # embeds: [batch, frames, data]
        # offsets: [batch, frames, 3]
        # bgroups: [batch, points, bgroups] should be same across batches during training

        batches = embeds.shape[0]
        frames = embeds.shape[1]
        n_points = bgroups.shape[1]

        if self.training:
            embeds = self.batch_dropout(embeds)

        embed_x = self.embed_in(torch.cat([embeds, offsets], dim=-1)).transpose(0, 1)

        for block in self.embed_attn:
            embed_x = block(embed_x)

        embed_x = embed_x.transpose(0, 1)

        x = self.bgroup_proj(embed_x)
        x = torch.reshape(x, (batches, frames, self.n_bgroups, -1))
        bgroups_idx = torch.argmax(bgroups, dim=-1)
        bgroups_idx = torch.broadcast_to(bgroups_idx.view(batches, 1, n_points, 1), (batches, frames, n_points, x.shape[-1]))
        x = torch.gather(x, -2, bgroups_idx)

        seed = torch.randn(batches, 1, n_points, self.d_seed, device=embeds.device)
        seed = self.point_proj(seed)
        
        x = x + seed
        x = x + self.output_mlp1(self.relu(x))
        x = x + self.output_mlp2(self.relu(x))
        x = x + self.output_mlp3(self.relu(x))

        out = self.output_proj(x)

        if apply_offset:
            out = out + offsets.unsqueeze(-2)

        bgroups = torch.broadcast_to(bgroups.unsqueeze(1), (-1, frames, -1, -1))
        out = torch.cat([out, bgroups], dim=-1)

        return out