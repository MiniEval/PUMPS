from data.skeleton import Skeleton
import numpy as np
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion

import torch
import glob
import os
import cdflib

from xml.etree import ElementTree as ET

DATASET_FOLDER = "../Human3.6M"
OUTPUT_FOLDER = "../processed/Human3.6M"

def parse_array(string, use_float=True):
    elem_list = string.strip("[]").split()
    if use_float:
        elem_list = [float(x) for x in elem_list]
    else:
        elem_list = [int(x) for x in elem_list]

    return elem_list

if __name__ == "__main__":
    data = ET.parse(DATASET_FOLDER + "/metadata.xml")

    root = data.getroot()
    data = root.find("skel_angles").find("tree").findall("item")
    joint_names = []
    joint_hierarchy = []
    joint_offsets = []
    end_sites = {}
    joint_id = 0
    parent_map = dict()
    parent_map[-1] = -1
    for i, joint in enumerate(data):
        name = joint.find("name").text
        parent = parent_map[int(joint.find("parent").text) - 1]
        offset = parse_array(joint.find("offset").text)
        parent_map[i] = joint_id
        if name == "Site":
            end_sites[joint_names[parent]] = offset
        else:
            joint_names.append(name)
            joint_offsets.append(offset)
            joint_hierarchy.append(parent)
            joint_id += 1

    children = [list() for _ in range(len(joint_names))]
    for i, parent_idx in enumerate(joint_hierarchy):
        if parent_idx >= 0:
            children[parent_idx].append(i)

    joint_tails = []

    joint_bgroups = torch.zeros(len(joint_names), 5, dtype=torch.float32)
    bgroups_spine = ["Hips", "Spine", "Spine1", "Neck", "Head"]
    bgroups_leftarm = ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumb", "L_Wrist_End"]
    bgroups_rightarm = ["RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumb", "R_Wrist_End"]
    bgroups_leftleg = ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"]
    bgroups_rightleg = ["RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]

    end_joints = ["LeftHand", "RightHand", "LeftToeBase", "RightToeBase"]

    for i, name in enumerate(joint_names):
        if len(children[i]) == 0:
            joint_tails.append(end_sites[name])
        elif name in ("Hips", "LeftHand", "RightHand"):
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

    joint_offsets = torch.tensor(joint_offsets, dtype=torch.float32)
    joint_tails = torch.tensor(joint_tails, dtype=torch.float32)

    pairs = []
    for i in range(len(bgroups_leftarm)):
        pairs.append((joint_names.index(bgroups_leftarm[i]), joint_names.index(bgroups_rightarm[i])))

    for i in range(len(bgroups_leftleg)):
        pairs.append((joint_names.index(bgroups_leftleg[i]), joint_names.index(bgroups_rightleg[i])))

    skeleton = Skeleton(joint_names, joint_hierarchy, joint_offsets, joint_tails, joint_bgroups, end_joints, pairs,
                        scale=0.01)
    
    filenames = sorted(glob.glob(DATASET_FOLDER + "/**/*.cdf", recursive=True))
    for file in filenames:
        print("Processing file:", file, end="\r")
        cdf_data = cdflib.CDF(file)
        raw_data = torch.tensor(cdf_data.varget("Pose")[0], dtype=torch.float32)
        
        data = raw_data[1::2].clone()
        # rotation_euler = torch.roll(torch.reshape(data[..., 3:], (data.shape[0], -1, 3)), 2, dims=-1)
        rotation_euler = torch.reshape(data[..., 3:], (data.shape[0], -1, 3))
        rotation_matrix = euler_angles_to_matrix(torch.deg2rad(rotation_euler), "ZXY")
        q = matrix_to_quaternion(rotation_matrix)

        sign = torch.gt(torch.linalg.vector_norm(q[1:] - q[:-1], dim=-1, keepdim=True),
                        torch.linalg.vector_norm(q[1:] + q[:-1], dim=-1, keepdim=True)).int()
        sign = (torch.cumsum(sign, dim=0) % 2) * -2 + 1
        q = torch.cat([q[:1], q[1:] * sign], dim=0)
        q = torch.reshape(q, (data.shape[0], -1))
        p, q = skeleton.fk(data[..., :3].unsqueeze(0) * 0.001, q.unsqueeze(0), local_q=True)
        q = torch.reshape(q, (data.shape[0], -1))

        processed_data = torch.cat([data[..., :3] * 0.001, q], dim=-1)
        output_motion_file = file.replace(DATASET_FOLDER, OUTPUT_FOLDER).replace(".cdf", ".pt")
        output_skeleton_file = file.replace(DATASET_FOLDER, OUTPUT_FOLDER).replace(".cdf", ".skel.pt")
        os.makedirs(output_motion_file.rsplit("/", maxsplit=1)[0], exist_ok=True)
        
        torch.save(processed_data, output_motion_file)
        torch.save(skeleton, output_skeleton_file)