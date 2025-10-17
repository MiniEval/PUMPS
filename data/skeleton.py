import torch
import numpy as np


class Skeleton:
    def __init__(self, joint_names, joint_hierarchy, joint_offsets, joint_tails, joint_bgroups, end_joints, pairs,
                 scale=1.0, device="cpu"):
        self.n_joints = len(joint_names)

        # 1d array, -1 for root
        self.joint_names = joint_names
        self.joint_hierarchy = joint_hierarchy
        # [left hand, right hand, left toe, right toe]
        self.end_joint_names = end_joints
        self.end_joints = torch.tensor([self.joint_names.index(name) for name in end_joints],
                                       dtype=torch.int64, device=device)

        # [j, 3] tensor
        self.joint_offsets = joint_offsets.clone().to(device) * scale
        self.joint_tails = joint_tails.clone().to(device) * scale

        # [j, n_bgroups] one-hot tensor
        self.joint_bgroups = joint_bgroups.clone().to(device)

        # clip skeleton at end joints & normalise end joint lengths
        joint_lengths = torch.linalg.vector_norm(self.joint_tails, dim=-1)
        
        for joint in self.end_joints:
            if joint_lengths[joint] < 1e-6:
                self.joint_tails[joint] = self.joint_tails[self.joint_hierarchy[joint]]
            self.joint_tails[joint] /= torch.linalg.vector_norm(self.joint_tails[joint])
            self.joint_tails[joint] *= 0.1

        # joint size for point cloud distribution
        self.joint_lengths = torch.linalg.vector_norm(self.joint_tails, dim=-1)
        self.joint_dist = torch.distributions.Categorical(probs=self.joint_lengths.cpu())

        # disable relative roll when >2 child joints
        n_children = np.zeros(len(self.joint_hierarchy))
        for i in self.joint_hierarchy[1:]:
            n_children[i] += 1
        self.roll_mask = torch.tensor([n_children[j] < 2 for j in range(len(self.joint_hierarchy))],
                                      dtype=torch.bool, device=device)
        self.pairs = torch.tensor(pairs, dtype=torch.int64, device=device)

        self.id_mat = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=device, requires_grad=False)

    def fk(self, root_p, q, local_q=True):
        # root_p: [b, l, 3]
        # q: [b, l, j, 4] or [b, l, j * 4]
        # output: [b, l, j, *]
        q = torch.reshape(q, (q.shape[0], q.shape[1], -1, 4))
        q = q / torch.clamp_min(torch.linalg.vector_norm(q, dim=-1, keepdim=True), 1e-5)

        if local_q:
            heads = []
            glob_q = []
            for i in range(len(self.joint_hierarchy)):
                parent_index = self.joint_hierarchy[i]
                if parent_index == -1:
                    head = root_p
                    _glob_q = q[:, :, i].contiguous()
                else:
                    head = self._qrot(self.joint_offsets[i], glob_q[parent_index])
                    head = head + heads[parent_index].clone().detach()
                    # _glob_q = self._qmul(glob_q[parent_index].clone().detach(), q[:, :, i].contiguous())
                    _glob_q = self._qmul(glob_q[parent_index], q[:, :, i].contiguous())

                heads.append(head)
                glob_q.append(_glob_q)

            heads = torch.stack(heads, dim=2)
            glob_q = torch.stack(glob_q, dim=2)

        else:
            heads = []
            parent_q = q.transpose(0, 2)[self.joint_hierarchy[1:]].transpose(0, 2)
            offsets = self._qrot(self.joint_offsets[1:], parent_q)
            for i in range(len(self.joint_hierarchy)):
                parent_index = self.joint_hierarchy[i]
                if parent_index == -1:
                    head = root_p
                else:
                    head = offsets[:, :, i - 1] + heads[parent_index]
                heads.append(head)

            heads = torch.stack(heads, dim=2)
            glob_q = q

        return heads, glob_q

    def keypoint_ik(self, root_p, keypoints, roll_offset=None, eps=1e-6):
        # root_p: [b, l, 3]
        # keypoints: [b, l, j, 5], joint tails (global) & roll
        # roll_offset: [b, j]
        # output: [b, l, j, *]

        root_p = torch.clone(root_p, memory_format=torch.contiguous_format)
        keypoints = torch.clone(keypoints, memory_format=torch.contiguous_format)

        directions = keypoints[..., :3]
        rolls = keypoints[..., 3:] # cos, sin order
        rolls = rolls / torch.clamp_min(torch.linalg.vector_norm(rolls, dim=-1, keepdim=True), eps)
        if roll_offset is not None:
            roll_offset = torch.clone(roll_offset, memory_format=torch.contiguous_format)
            roll_offset = (roll_offset * self.roll_mask).unsqueeze(1)

            # complex 2d rotation
            sin_offset = torch.sin(roll_offset)
            cos_offset = torch.cos(roll_offset)
            rolls = torch.stack([cos_offset * rolls[..., 0] - sin_offset * rolls[..., 1],
                                 sin_offset * rolls[..., 0] + cos_offset * rolls[..., 1]], dim=-1)

        heads = []
        q = []
        for i in range(len(self.joint_hierarchy)):
            parent_index = self.joint_hierarchy[i]
            if parent_index == -1:
                head = root_p
            else:
                head = self._qrot(self.joint_offsets[i], q[parent_index])
                head = head + heads[parent_index]

            kp = directions[..., i, :].clone() - head
            q_ = self.batch_single_ik(torch.reshape(self.joint_tails[i], (1, 1, -1)), kp, rolls[..., i, :], eps=eps)

            heads.append(head)
            q.append(q_)

        # heads, tails: [b, l, j, 3], q: [b, l, j, 4]
        heads = torch.stack(heads, dim=2).clone()
        q = torch.stack(q, dim=2).clone()
        return heads, q

    def batch_single_ik(self, raw_u, raw_v, roll, eps=1e-6):
        # u: [..., 3] or [..., l, 3], rest
        # v: [..., l, 3], target
        # roll: [..., l, 2], cos + sin half-angle values

        raw_u = torch.clone(raw_u, memory_format=torch.contiguous_format)
        raw_v = torch.clone(raw_v, memory_format=torch.contiguous_format)
        roll = torch.clone(roll, memory_format=torch.contiguous_format)

        with torch.no_grad():
            raw_u_check = torch.linalg.vector_norm(raw_u, dim=-1)
            raw_u[raw_u_check < eps] = self.id_mat[0, :3]

            raw_v_check = torch.linalg.vector_norm(raw_v, dim=-1)
            raw_v[raw_v_check < eps] = self.id_mat[0, :3]

            roll_check = torch.linalg.vector_norm(roll, dim=-1)
            roll[roll_check < eps] = self.id_mat[0, :2]

        u = raw_u / torch.linalg.vector_norm(raw_u, dim=-1, keepdim=True)
        v = raw_v / torch.linalg.vector_norm(raw_v, dim=-1, keepdim=True)
        if len(raw_u.shape) != len(raw_v.shape):
            u = u.unsqueeze(-2)

        median = (u + v) / 2
        raw_q = torch.cat([torch.full((*median.shape[:-1], 1), eps, device=median.device), median], dim=-1)
        raw_q = raw_q / torch.linalg.vector_norm(raw_q, dim=-1, keepdim=True)

        roll = roll / torch.linalg.vector_norm(roll, dim=-1, keepdim=True)
        cos_roll = roll[..., :1]
        sin_roll = roll[..., 1:]

        roll_q = torch.cat([cos_roll, sin_roll * u], dim=-1)

        q = self._qmul(raw_q, roll_q)
        q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)

        return q

    def q_to_keypoint(self, q, rest, eps=1e-6):
        # rest: [..., 3]
        # q: [..., l, 4]

        q = q.clone()
        rest = rest.clone()

        with torch.no_grad():
            q_check = torch.linalg.vector_norm(q, dim=-1)
            q[q_check < eps] = self.id_mat

            rest_check = torch.linalg.vector_norm(rest, dim=-1)
            rest[rest_check < eps] = self.id_mat[0, :3]

        u = rest / torch.linalg.vector_norm(rest, dim=-1, keepdim=True)
        u = u.unsqueeze(-2)
        v = self._qrot(u, q)

        median = (u + v) / 2
        raw_q = torch.cat([torch.full((*median.shape[:-1], 1), eps, device=median.device), median], dim=-1)
        raw_q = raw_q / torch.linalg.vector_norm(raw_q, dim=-1, keepdim=True)
        roll_q = self._qmul(-raw_q, q)

        cos_roll = roll_q[..., :1]

        sin_roll = torch.where(torch.abs(u) > eps, roll_q[..., 1:], 0) / (torch.where(torch.abs(u) > eps, u, 1))
        sin_roll_idx = torch.argmax(torch.abs(sin_roll), dim=-1, keepdim=True)
        sin_roll = torch.gather(sin_roll, -1, sin_roll_idx)

        roll = torch.cat([cos_roll, sin_roll], dim=-1)

        return torch.cat([v, roll], dim=-1)

    @staticmethod
    def _qrot(p, q):
        # p: [..., 3], broadcastable
        # q: [..., 4], broadcastable
        if len(p.shape) < len(q.shape):
            shape = [1 for _ in range(len(q.shape) - len(p.shape))]
            shape = [*shape, *p.shape]
            p = torch.reshape(p, shape)
        elif len(q.shape) < len(p.shape):
            shape = [1 for _ in range(len(p.shape) - len(q.shape))]
            shape = [*shape, *q.shape]
            q = torch.reshape(q, shape)

        t = 2 * torch.linalg.cross(q[..., 1:], p, dim=-1)
        new_p = p + (q[..., :1] * t) + torch.linalg.cross(q[..., 1:], t, dim=-1)
        return new_p

    @staticmethod
    def _qmul(u, v):
        # u, v: [..., 4], same dim, real first

        original_shape = u.shape
        terms = torch.bmm(torch.reshape(v, (-1, 4, 1)), torch.reshape(u, (-1, 1, 4)))

        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.reshape(torch.stack((w, x, y, z), dim=1), original_shape).contiguous()

    @staticmethod
    def _qinv(q):
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    def generate_pointcloud(self, global_p, global_q, n_points=256, std=1.0, output_joints=False):
        # global_p: [b, l, j, 3]
        # global_q: [b, l, j, 4]
        # output: [b, l, n, 3 + bgroups]

        batches = global_p.shape[0]
        frames = global_p.shape[1]

        joints = self.joint_dist.sample(torch.Size((n_points,))).to(global_p.device)
        origins = torch.rand((batches, n_points), dtype=torch.float32, device=global_p.device)
        offsets = torch.normal(torch.zeros((batches, n_points, 3), dtype=torch.float32, device=global_p.device), std)

        tails_sample = torch.reshape(self.joint_tails[joints], (1, 1, -1, 3)).clone()

        indices = torch.reshape(joints, (1, 1, -1, 1))
        heads_sample = torch.gather(global_p, 2, indices.repeat(batches, frames, 1, 3))
        q_sample = torch.gather(global_q, 2, indices.repeat(batches, frames, 1, 4))

        bgroups = torch.reshape(self.joint_bgroups[joints].clone(), (1, 1, n_points, -1))

        # 1. normal offset position origin along rest position of joint
        pointcloud = tails_sample * torch.reshape(origins, (batches, 1, -1, 1))
        # 2. apply normal offset
        pointcloud = pointcloud + offsets.unsqueeze(1)
        # 3. apply current global rotation of joint to get parent-local point position
        pointcloud = self._qrot(pointcloud, q_sample)
        # 4. apply parent joint position to get global point position
        pointcloud = pointcloud + heads_sample
        # 5. append body group data
        pointcloud = torch.cat([pointcloud, bgroups.repeat(batches, frames, 1, 1)], dim=-1)

        if output_joints:
            return pointcloud, joints
        else:
            return pointcloud

    def get_tails(self, global_p, global_q):
        # global_p & global_q: [..., j, 4]
        tail_offset = self._qrot(self.joint_tails, global_q)
        tails = global_p + tail_offset

        return tails

    @staticmethod
    def _slerp(poses, key_mask):
        # poses: [batch (optional), frames, data]
        # key_mask: [batch (optional), frames], keyframe = True

        if len(poses.shape) == 2:
            poses_ = poses.unsqueeze(0)
            key_mask_ = key_mask.unsqueeze(0)
        else:
            poses_ = poses
            key_mask_ = key_mask

        batches = poses_.shape[0]
        frames = poses_.shape[1]

        p = poses_[..., :3]
        q = torch.reshape(poses_[..., 3:], (batches, frames, -1, 4))
        interp = []
        for b in range(poses.shape[0]):
            keys = torch.nonzero(key_mask_[b])[..., 0].tolist()
            p_lerp = [p[b, :1]]
            q_slerp = [q[b, :1]]
            for i in range(len(keys) - 1):
                start_f = keys[i]
                end_f = keys[i+1]
                interval = end_f - start_f
                start_p = p[b, start_f]
                end_p = p[b, end_f]
                start_q = q[b, start_f]
                end_q = q[b, end_f]

                # lerp for root position
                weight = (torch.arange(interval, dtype=torch.float32, device=poses.device) + 1) / interval
                p_lerp.append(start_p * (1 - weight.view(-1, 1)) + end_p * weight.view(-1, 1))

                # slerp for rotations
                q_cross = Skeleton._qmul(Skeleton._qinv(start_q), end_q)
                # numeric stability
                q_cross = q_cross / torch.clamp_min(torch.linalg.vector_norm(q_cross, dim=-1, keepdim=True), 1e-4)

                q_real = torch.cos(weight.view(-1, 1, 1) * torch.acos(q_cross[..., :1]))
                q_axis = q_cross[..., 1:] / torch.clamp_min(torch.linalg.vector_norm(q_cross[..., 1:], dim=-1, keepdim=True), 1e-4)
                q_axis = q_axis * torch.sqrt(1 - (q_real ** 2))
                q_rot = torch.cat([q_real, q_axis], dim=-1)

                q_slerp.append(Skeleton._qmul(torch.broadcast_to(start_q, q_rot.shape), q_rot))
            p_lerp = torch.cat(p_lerp, dim=0)
            q_slerp = torch.reshape(torch.cat(q_slerp, dim=0), (frames, -1))
            interp.append(torch.cat([p_lerp, q_slerp], dim=-1))

        return torch.stack(interp, dim=0)

    def to_device(self, device):
        if self.joint_tails.device != device:
            return Skeleton(self.joint_names, self.joint_hierarchy, self.joint_offsets, self.joint_tails,
                            self.joint_bgroups, self.end_joint_names, self.pairs.detach().cpu().numpy(), scale=1.0, device=device)
        return self


class SkeletonStack:
    def __init__(self, skeletons, device=torch.device('cpu')):
        self.skeletons = [skel.to_device(device) for skel in skeletons]
        self.joint_offsets = torch.stack([s.joint_offsets for s in self.skeletons], dim=0)
        self.joint_tails = torch.stack([s.joint_tails for s in self.skeletons], dim=0)
        self.joint_dist = self.skeletons[0].joint_dist # first skeleton dist only due to parallelisation
        self.joint_hierarchy = self.skeletons[0].joint_hierarchy
        self.joint_bgroups = self.skeletons[0].joint_bgroups
        self.n_joints = self.skeletons[0].n_joints

    def fk(self, root_p, q, local_q=True):
        # batches should align with number of skeletons
        # root_p: [b, l, 3]
        # q: [b, l, j, 4] or [b, l, j * 4]
        # output: [b, l, j, *]
        q = torch.reshape(q, (q.shape[0], q.shape[1], -1, 4))
        q = q / torch.clamp_min(torch.linalg.vector_norm(q, dim=-1, keepdim=True), 1e-5)

        if local_q:
            heads = []
            glob_q = []
            for i in range(len(self.joint_hierarchy)):
                parent_index = self.joint_hierarchy[i]
                if parent_index == -1:
                    head = root_p
                    _glob_q = q[:, :, i].contiguous()
                else:
                    head = Skeleton._qrot(self.joint_offsets[:, i:i+1], glob_q[parent_index])
                    head = head + heads[parent_index].clone().detach()
                    _glob_q = Skeleton._qmul(glob_q[parent_index].clone().detach(), q[:, :, i].contiguous())
                    # if i == 13 or i == 14:
                    #     _glob_q = self.skeletons[0]._qmul(glob_q[6], q[:, :, i].contiguous())
                    # else:
                    #     _glob_q = self.skeletons[0]._qmul(glob_q[parent_index], q[:, :, i].contiguous())

                heads.append(head)
                glob_q.append(_glob_q)

            heads = torch.stack(heads, dim=2)
            glob_q = torch.stack(glob_q, dim=2)

        else:
            heads = []
            parent_q = q.transpose(0, 2)[self.joint_hierarchy[1:]].transpose(0, 2)
            # print(self.joint_offsets.shape, parent_q.shape)
            offsets = Skeleton._qrot(self.joint_offsets[:, 1:].unsqueeze(-3), parent_q)
            for i in range(len(self.joint_hierarchy)):
                parent_index = self.joint_hierarchy[i]
                if parent_index == -1:
                    head = root_p
                else:
                    head = offsets[:, :, i - 1] + heads[parent_index]
                heads.append(head)

            heads = torch.stack(heads, dim=2)
            glob_q = q

        return heads, glob_q

    def generate_pointcloud(self, global_p, global_q, n_points=256, std=1.0, output_joints=False):
        # global_p: [b, l, j, 3]
        # global_q: [b, l, j, 4]
        # output: [b, l, n, 3 + bgroups]
        # std: [b, n, 3] broadcastable

        batches = global_p.shape[0]
        frames = global_p.shape[1]

        joints = self.joint_dist.sample(torch.Size((n_points,))).to(global_p.device)
        origins = torch.rand((batches, n_points), dtype=torch.float32, device=global_p.device)
        offsets = torch.normal(torch.zeros((batches, n_points, 3), dtype=torch.float32, device=global_p.device), std)

        tails_sample = torch.reshape(self.joint_tails.transpose(0, 1)[joints].transpose(0, 1), (batches, 1, -1, 3)).clone()

        indices = torch.reshape(joints, (1, 1, -1, 1))
        heads_sample = torch.gather(global_p, 2, indices.repeat(batches, frames, 1, 3))
        q_sample = torch.gather(global_q, 2, indices.repeat(batches, frames, 1, 4))

        bgroups = torch.reshape(self.joint_bgroups[joints].clone(), (1, 1, n_points, -1))

        # 1. normal offset position origin along rest position of joint
        pointcloud = tails_sample * torch.reshape(origins, (batches, 1, -1, 1))
        # 2. apply normal offset
        pointcloud = pointcloud + offsets.unsqueeze(1)
        # 3. apply current global rotation of joint to get parent-local point position
        pointcloud = Skeleton._qrot(pointcloud, q_sample)
        # 4. apply parent joint position to get global point position
        pointcloud = pointcloud + heads_sample
        # 5. append body group data
        pointcloud = torch.cat([pointcloud, bgroups.repeat(batches, frames, 1, 1)], dim=-1)

        if output_joints:
            return pointcloud, joints
        else:
            return pointcloud

    def get_tails(self, global_p, global_q):
        # global_p & global_q: [b, l, j, 4]
        tail_offset = Skeleton._qrot(self.joint_tails.unsqueeze(-3), global_q)
        tails = global_p + tail_offset

        return tails

    def to_device(self, device):
        if self.joint_tails.device != device:
            return SkeletonStack(self.skeletons, device)
        return self