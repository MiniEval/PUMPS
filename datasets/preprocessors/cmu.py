from dataset.preprocessors.bvh import parse_bvh_skeleton
from data.skeleton import Skeleton
import numpy as np
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion

import torch
import glob
import os

DATASET_FOLDER = "../CMU/data"
OUTPUT_FOLDER = "../processed/CMU"

if __name__ == "__main__":
    raw_files = glob.glob(DATASET_FOLDER + "/**/*.bvh", recursive=True)
    
    with open(raw_files[0], "r") as f:
        joint_names, joint_offsets, joint_hierarchy, end_sites = parse_bvh_skeleton(f.read())

    children = [list() for _ in range(len(joint_names))]
    for i, parent_idx in enumerate(joint_hierarchy):
        if parent_idx >= 0:
            children[parent_idx].append(i)

    joint_tails = []

    joint_bgroups = torch.zeros(len(joint_names), 5, dtype=torch.float32)
    bgroups_spine = ["Hips", "LHipJoint", "RHipJoint", "LowerBack", "Spine", "Spine1", "Neck", "Neck1", "Head"]
    bgroups_leftarm = ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftFingerBase", "LeftHandIndex1"]
    bgroups_rightarm = ["RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightFingerBase", "RightHandIndex1"]
    bgroups_leftleg = ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"]
    bgroups_rightleg = ["RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]

    end_joints = ["LeftHand", "RightHand", "LeftToeBase", "RightToeBase"]

    for i, name in enumerate(joint_names):
        if len(children[i]) == 0:
            joint_tails.append(end_sites[name])
        elif name == "Hips":
            joint_tails.append(joint_offsets[children[i][-1]])
        else:
            joint_tails.append(joint_offsets[children[i][0]])

        if name in bgroups_spine:
            joint_bgroups[i, 0] = 1
        elif name in bgroups_leftarm:
            joint_bgroups[i, 1] = 1
        elif name in bgroups_rightarm:
            joint_bgroups[i, 2] = 1
        elif name in bgroups_leftleg:
            joint_bgroups[i, 3] = 1
        elif name in bgroups_rightleg:
            joint_bgroups[i, 4] = 1

    pairs = []
    for i in range(len(bgroups_leftarm)):
        pairs.append((joint_names.index(bgroups_leftarm[i]), joint_names.index(bgroups_rightarm[i])))

    for i in range(len(bgroups_leftleg)):
        pairs.append((joint_names.index(bgroups_leftleg[i]), joint_names.index(bgroups_rightleg[i])))

    joint_offsets = torch.tensor(joint_offsets, dtype=torch.float32)
    joint_tails = torch.tensor(joint_tails, dtype=torch.float32)

    skeleton = Skeleton(joint_names, joint_hierarchy, joint_offsets, joint_tails, joint_bgroups, end_joints, pairs,
                        scale=0.056444)
    
    fps60 = ["33_", "34_", "42_", "60_", "61_", "62_", "74_", "75_", "77_", "79_", "80_", "87_", "88_", "89_"]
    filenames = sorted(glob.glob(DATASET_FOLDER + "/**/*.bvh", recursive=True))
    for file in filenames:
        print("Processing file:", file, end="\r")
        with open(file, "r") as f:
            while f.readline() != "MOTION\n":
                continue
            file_data = f.read()
        
        lines = file_data.split('\n')
        motion_data = []
        for line in lines[2:]:
            words = line.strip().split(" ")
            try:
                motion_data.append(list(map(float, words)))
            except ValueError:
                pass
        raw_data = torch.tensor(motion_data, dtype=torch.float32)
        
        if any(s in file for s in fps60):
            data = raw_data[1::2].clone()
        else:
            data = raw_data[1::4].clone()
        rotation_euler = torch.reshape(data[..., 3:], (data.shape[0], -1, 3))
        rotation_matrix = euler_angles_to_matrix(torch.deg2rad(rotation_euler), "ZYX")
        q = matrix_to_quaternion(rotation_matrix)

        _xzy_to_xyz = torch.sqrt(torch.tensor([[2, 2, 0, 0]], dtype=torch.float32)) / 2
        _xzy_to_xyz = _xzy_to_xyz.expand(q[:, 0].shape)
        q[:, 0] = Skeleton._qmul(_xzy_to_xyz, q[:, 0])
        data[..., :3] = Skeleton._qrot(data[..., :3], _xzy_to_xyz)

        sign = torch.gt(torch.linalg.vector_norm(q[1:] - q[:-1], dim=-1, keepdim=True),
                        torch.linalg.vector_norm(q[1:] + q[:-1], dim=-1, keepdim=True)).int()
        sign = (torch.cumsum(sign, dim=0) % 2) * -2 + 1
        q = torch.cat([q[:1], q[1:] * sign], dim=0)
        p, q = skeleton.fk(data[..., :3].unsqueeze(0) * 0.056444, q.unsqueeze(0), local_q=True)
        q = torch.reshape(q, (data.shape[0], -1))

        processed_data = torch.cat([data[..., :3] * 0.056444, q], dim=-1)
        output_motion_file = file.replace(DATASET_FOLDER, OUTPUT_FOLDER).replace(".bvh", ".pt")
        output_skeleton_file = file.replace(DATASET_FOLDER, OUTPUT_FOLDER).replace(".bvh", ".skel.pt")
        os.makedirs(output_motion_file.rsplit("/", maxsplit=1)[0], exist_ok=True)
        
        torch.save(processed_data, output_motion_file)
        torch.save(skeleton, output_skeleton_file)