import random

import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import bisect
from data.skeleton import SkeletonStack, Skeleton


class MotionDataset(Dataset):
    def __init__(self, data_folder, sample_length, device='cpu'):
        self.data = []
        self.device = device
        
        self.sample_length = sample_length

        skeleton_files = glob.glob(data_folder + "/**/*.skel.pt", recursive=True)
        motion_files = [s.replace(".skel.pt", ".pt") for s in skeleton_files]
        
        self.data_frames = []
        for i, skeleton_file in enumerate(skeleton_files):
            motion = torch.load(motion_files[i], weights_only=False)
            if motion.shape[0] < sample_length:
                continue
            skeleton = torch.load(skeleton_file, weights_only=False)
            self.data.append((motion, skeleton))
            self.data_frames.append(motion.shape[0])
        
        self.sample_weights = np.cumsum(self.data_frames) / np.sum(self.data_frames)
        print("Loaded %d motions from data folder %s" % (len(self.data), data_folder))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # point_std: [samples, points, 3], broadcastable
        r = random.random()
        idx = bisect.bisect_left(self.sample_weights, r)

        motion, skeleton = self.data[idx]
        start = random.randint(0, motion.shape[0] - self.sample_length)
        return motion[start:start + self.sample_length], skeleton
    
    def preprocess_samples(self, motions, skeletons, random_z=True, use_fk=False, 
                           use_points=False, n_points=256, point_std=0.05):
        n_samples = len(motions)
        
        motions = torch.stack(motions, dim=0).to(self.device)
        motions = self.normalise_motion(motions, skeletons, mode="root")
        skeletons = SkeletonStack(skeletons, device=self.device)
        
        if random_z:
            p, q = skeletons.fk(motions[..., :3], motions[..., 3:], local_q=False)
            z_aug = 1 - torch.rand(n_samples)
            z_aug = torch.reshape(z_aug, (n_samples, 1))
            z_aug = torch.cat([z_aug, torch.zeros(n_samples, 2), torch.sqrt(1 - (z_aug ** 2))], dim=-1)
            z_aug = torch.reshape(z_aug, (n_samples, 1, 1, 4)).to(self.device)
            z_aug = torch.broadcast_to(z_aug, (n_samples, self.sample_length, skeletons.n_joints, 4))
            
            p = Skeleton._qrot(p, z_aug)
            q = Skeleton._qmul(z_aug, q)
            motions = torch.cat([p[..., 0, :], torch.reshape(q, (n_samples, self.sample_length, -1))], dim=-1)
        
        ret = [motions, skeletons]
        
        if use_fk or use_points:
            p, q = skeletons.fk(motions[..., :3], motions[..., 3:], local_q=False)
            if use_fk:
                ret.extend([p, q])
            if use_points:
                points = skeletons.generate_pointcloud(p, q, n_points=n_points, std=point_std)
                ret.append(points)
        
        return tuple(ret)
    
    @staticmethod
    def normalise_motion(motions, skeletons=None, mode="root", return_mean=False, random_displace=1.0):
        # motion: [batches, frames, data]
        # modes: ["root", "bones"]
        
        uniform = (torch.rand(motions.shape[0], 1, 3, device=motions.device) - 0.5) * random_displace
        
        if mode == "root":
            means = torch.mean(motions[..., :3], dim=-2, keepdim=True)
        elif mode == "bones" and skeletons is not None:
            p, q = skeletons.fk(motions[..., :3], motions[..., 3:], local_q=False)
            tails = skeletons.get_tails(p, q)
            means = torch.mean(torch.cat([p, tails], dim=-2), dim=(-2, -3)).unsqueeze(-2)
        elif mode == "bones":
            raise ValueError("Skeletons are required for bone normalisation.")
        else:
            raise ValueError("Invalid mode:", mode)
        
        means = means + uniform
        normalised = torch.cat([motions[..., :3] - means, motions[..., 3:]], dim=-1)
        if return_mean:
            return normalised, means
        else:
            return normalised
    
    @staticmethod
    def normalise_points(points, return_mean=False, random_displace=1.0):
        # points: [batches, frames, points, 3 + optional bodygroups]
        
        uniform = (torch.rand(points.shape[0], 1, 1, 3, device=points.device) - 0.5) * random_displace
        
        means = torch.mean(points[..., :3], dim=(-2, -3), keepdim=True) + uniform
        normalised = torch.cat([points[..., :3] - means, points[..., 3:]], dim=-1)
        if return_mean:
            return normalised, means
        else:
            return normalised
        
    def collate_motion_points(self, batch, MIN_STD=0.025, MAX_STD=0.075):
        motions = []
        skeletons = []
        for m, s in batch:
            motions.append(m)
            skeletons.append(s)
        std = torch.rand(len(motions), 1, 1, device=self.device) * (MAX_STD - MIN_STD) + MIN_STD
        _, _, points = self.preprocess_samples(motions, skeletons, use_fk=False, use_points=True, point_std=std)
        return points
