import numpy as np
import glob
import torch
from pytorch3d.transforms import axis_angle_to_quaternion
from data.skeleton import Skeleton
import os
import math


DATASET_FOLDER = "../AMASS"
OUTPUT_FOLDER = "../processed/AMASS"
SMPL_PATH = "../smpl"

SMPLH_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    # 'left_index1',
    # 'left_index2',
    # 'left_index3',
    # 'left_middle1',
    # 'left_middle2',
    # 'left_middle3',
    # 'left_pinky1',
    # 'left_pinky2',
    # 'left_pinky3',
    # 'left_ring1',
    # 'left_ring2',
    # 'left_ring3',
    # 'left_thumb1',
    # 'left_thumb2',
    # 'left_thumb3',
    # 'right_index1',
    # 'right_index2',
    # 'right_index3',
    # 'right_middle1',
    # 'right_middle2',
    # 'right_middle3',
    # 'right_pinky1',
    # 'right_pinky2',
    # 'right_pinky3',
    # 'right_ring1',
    # 'right_ring2',
    # 'right_ring3',
    # 'right_thumb1',
    # 'right_thumb2',
    # 'right_thumb3'
]

def ensure_str(s, encoding='utf-8'):
    if isinstance(s, bytes):
        return s.decode(encoding)
    elif isinstance(s, str):
        return s
    else:
        raise TypeError(f"Expected str or bytes, got {type(s)}")

if __name__ == "__main__":
    filenames = glob.glob("/datasets/AMASS/**/*.npz", recursive=True)
    
    n_body_joints = 22
    
    for filename in filenames:
        data = np.load(filename, allow_pickle=True)
        if(len(dict(data).keys()) < 6 or "poses" not in dict(data).keys()):
            continue
        
        print("Processing file:", filename, end="\r")
        
        betas = torch.tensor(data["betas"], dtype=torch.float32).unsqueeze(0)
        poses = torch.tensor(data['poses'], dtype=torch.float32)
        trans = torch.tensor(data['trans'], dtype=torch.float32)
        gender = ensure_str(data["gender"].item())
        framerate_keys = ["mocap_framerate", "mocap_frame_rate"]
        framerate = 30
        try:
            framerate_key = next(key for key in framerate_keys if key in data)
            if data[framerate_key] is not None:
                framerate = data[framerate_key]
        except:
            print("framerate not found in", filename)
        step = round(framerate / 30)
        
        body = np.load("%s/%s/model.npz" % (SMPL_PATH, gender))
        joint_hierarchy = body["kintree_table"][0, :n_body_joints]
        joint_hierarchy[0] = -1
        
        v_shaped = body["v_template"] + np.einsum('aib,jb->ai', body["shapedirs"], betas)
        joint_offsets = body["J_regressor"].dot(v_shaped)
        joint_offsets = torch.tensor(joint_offsets[:n_body_joints], dtype=torch.float32)
        joint_offsets[1:] -= joint_offsets[joint_hierarchy[1:]]
        
        poses = poses.reshape(poses.shape[0], -1, 3)
        quaternions = axis_angle_to_quaternion(poses[:, :n_body_joints])
        
        children = [list() for _ in range(len(joint_hierarchy))]
        for i, parent_idx in enumerate(joint_hierarchy):
            if parent_idx >= 0:
                children[parent_idx].append(i)

        joint_tails = []
        
        joint_bgroups = torch.zeros(len(SMPLH_JOINT_NAMES), 5, dtype=torch.float32)
        bgroups_spine = ["pelvis", "spine1", "spine2", "neck", "head"]
        bgroups_leftarm = ["left_collar", "left_shoulder", "left_elbow", "left_wrist"]
        bgroups_rightarm = ["right_collar", "right_shoulder", "right_elbow", "right_wrist"]
        bgroups_leftleg = ["left_hip", "left_knee", "left_ankle", "left_foot"]
        bgroups_rightleg = ["right_hip", "right_knee", "right_ankle", "right_foot"]
        END_JOINTS = ["left_wrist", "right_wrist", "left_foot", "right_foot"]
        
        for i, name in enumerate(SMPLH_JOINT_NAMES):
            if len(children[i]) == 0:
                joint_tails.append(joint_tails[joint_hierarchy[i]])
            elif name in ("spine3"):
                joint_tails.append(joint_offsets[children[i][0]])
            else:
                joint_tails.append(joint_offsets[children[i][-1]])
            
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
        joint_tails = torch.tensor(np.stack(joint_tails, axis=0), dtype=torch.float32)
        
        pairs = []
        for i in range(len(bgroups_leftarm)):
            pairs.append((SMPLH_JOINT_NAMES.index(bgroups_leftarm[i]), SMPLH_JOINT_NAMES.index(bgroups_rightarm[i])))

        for i in range(len(bgroups_leftleg)):
            pairs.append((SMPLH_JOINT_NAMES.index(bgroups_leftleg[i]), SMPLH_JOINT_NAMES.index(bgroups_rightleg[i])))
        
        skeleton = Skeleton(SMPLH_JOINT_NAMES, joint_hierarchy, joint_offsets, joint_tails, joint_bgroups, END_JOINTS, pairs)
        p, q = skeleton.fk(trans.unsqueeze(0), quaternions.unsqueeze(0), local_q=True)
        q = torch.reshape(q, (quaternions.shape[0], -1))
        
        processed_data = torch.cat([trans, q], dim=-1)[::step]
        output_motion_file = filename.replace(DATASET_FOLDER, OUTPUT_FOLDER).replace(".npz", ".pt")
        output_skeleton_file = filename.replace(DATASET_FOLDER, OUTPUT_FOLDER).replace(".npz", ".skel.pt")
        os.makedirs(output_motion_file.rsplit("/", maxsplit=1)[0], exist_ok=True)
        torch.save(processed_data, output_motion_file)
        torch.save(skeleton, output_skeleton_file)