import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from data.dataloader import MotionDataset
from models.MotionTransformer import MotionTransformer
from models.PoseEncoder import PoseEncoder, PointCloudDecoder

EPOCHS = 100000
START_EPOCH = 0
BATCH_SIZE = 16 # per dataset
N_FRAMES = 128
MIN_STD = 0.025
MAX_STD = 0.075

class MotionTrainer:
    def __init__(self, motion_model, point_encoder):
        self.motion_model = motion_model
        self.motion_model.train()
        self.point_encoder = point_encoder
        self.point_encoder.eval()
    
    def parameters(self):
        return self.motion_model.parameters()

    @staticmethod
    def generate_mask(batches, frames, mask_type="interp", device="cpu"):
        mask = torch.ones(batches, frames, dtype=torch.bool) # 1 = key, 0 = missing

        if mask_type == "interp":
            n_masked = frames - random.randint(frames // 24, frames // 4)
            for b in range(batches):
                masked_idx = torch.randperm(frames - 2)[:n_masked] + 1
                mask[b][masked_idx] = 0
        elif mask_type == "transition":
            n_masked = frames - random.randint(frames // 12, frames // 4)
            for b in range(batches):
                start = random.randint(1, frames - n_masked - 1)
                mask[b, start:start + n_masked] = 0
        elif mask_type == "predict":
            n_masked = frames - random.randint(frames // 12, frames // 4)
            for b in range(batches):
                if random.random() < 0.5:
                    # mask left side
                    mask[b, :n_masked] = 0
                else:
                    # mask right side
                    mask[b, -n_masked:] = 0
        else:
            raise TypeError("Unknown mask type, must be \"interp\", \"transition\", or \"predict\"")
        
        return mask.to(device)

    def train_masked_step(self, point_clouds):
        # point_clouds: [batch, frames, points, 3 + bgroups]
        batches = point_clouds.shape[0]
        frames = point_clouds.shape[1]
        n_points = point_clouds.shape[2]
        ret = dict()

        with torch.no_grad():
            embeds, offsets = self.point_encoder(torch.reshape(point_clouds, (batches * frames, n_points, -1)), return_offset=True)
            embeds = torch.reshape(embeds, (batches, frames, -1))
            offsets = torch.reshape(offsets, (batches, frames, 3))

            mask_interp = self.generate_mask(batches, frames, "interp", embeds.device)
            mask_transition = self.generate_mask(batches, frames, "transition", embeds.device)
            mask_predict = self.generate_mask(batches, frames, "predict", embeds.device)
        
        embeds_interp, offset_interp = self.motion_model(embeds, offsets, mask_interp)
        embeds_transition, offset_transition = self.motion_model(embeds, offsets, mask_transition)
        embeds_predict, offset_predict = self.motion_model(embeds, offsets, mask_predict)

        loss_embeds_interp = F.mse_loss(embeds_interp, embeds)
        loss_embeds_transition = F.mse_loss(embeds_transition, embeds)
        loss_embeds_predict = F.mse_loss(embeds_predict, embeds)
        loss_offset_interp = F.mse_loss(offset_interp, offsets)
        loss_offset_transition = F.mse_loss(offset_transition, offsets)
        loss_offset_predict = F.mse_loss(offset_predict, offsets)

        loss = loss_embeds_interp + loss_embeds_predict + loss_embeds_transition + \
               loss_offset_interp + loss_offset_predict + loss_offset_transition
        loss.backward()

        ret["point_clouds"] = point_clouds.clone().detach()
        ret["embeds"] = embeds.clone().detach()
        ret["offsets"] = offsets.clone().detach()

        ret["mask_interp"] = mask_interp.clone().detach()
        ret["mask_transition"] = mask_transition.clone().detach()
        ret["mask_predict"] = mask_predict.clone().detach()
        ret["embeds_interp"] = embeds_interp.clone().detach()
        ret["embeds_transition"] = embeds_transition.clone().detach()
        ret["embeds_predict"] = embeds_predict.clone().detach()
        ret["offset_interp"] = offset_interp.clone().detach()
        ret["offset_transition"] = offset_transition.clone().detach()
        ret["offset_predict"] = offset_predict.clone().detach()

        ret["loss_embeds_interp"] = loss_embeds_interp.item()
        ret["loss_embeds_transition"] = loss_embeds_transition.item()
        ret["loss_embeds_predict"] = loss_embeds_predict.item()
        ret["loss_offset_interp"] = loss_offset_interp.item()
        ret["loss_offset_transition"] = loss_offset_transition.item()
        ret["loss_offset_predict"] = loss_offset_predict.item()
        
        return ret

    def save_ckpt(self, n_iter, optimiser=None):
        subfolder = "./checkpoints/motion%d/" % n_iter
        os.makedirs(subfolder, exist_ok=True)

        torch.save(self.motion_model.module.state_dict(), subfolder + 'model.pt')
        if optimiser:
            torch.save(optimiser.state_dict(), subfolder + 'optimiser.pt')


def get_dataloader(rank, world_size):
    dataset_h36m = MotionDataset("/datasets/processed/Human3.6M", sample_length=N_FRAMES, device=rank)
    dataset_style = MotionDataset("/datasets/processed/100STYLE", sample_length=N_FRAMES, device=rank)
    dataset_lafan = MotionDataset("/datasets/processed/LaFAN1", sample_length=N_FRAMES, device=rank)
    dataset_cmu = MotionDataset("/datasets/processed/CMU", sample_length=N_FRAMES, device=rank)
    dataset_amass = MotionDataset("/datasets/processed/AMASS", sample_length=N_FRAMES, device=rank)
    datasets = [dataset_h36m, dataset_lafan, dataset_style, dataset_cmu, dataset_amass]
    dataloaders = []
    for dataset in datasets:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloaders.append(DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                      collate_fn=dataset.collate_motion_points))
    return dataloaders


def create_trainer(rank, encoder_ckpt, motion_model_ckpt=None):
    encoder = PoseEncoder(n_bgroups=5, d_model=32, k=8).to(rank)
    encoder.load_state_dict(torch.load(encoder_ckpt))
    motion_model = MotionTransformer(d_input=512 + 3,
                                     d_model=512,
                                     d_output=512 + 3,
                                     n_heads=8,
                                     n_layers=8,
                                     max_length=N_FRAMES).to(rank)
    if motion_model_ckpt is not None:
        motion_model.load_state_dict(torch.load(motion_model_ckpt))

    motion_model = DDP(motion_model, device_ids=[rank])

    return MotionTrainer(motion_model, encoder)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train(rank, world_size):
    setup(rank, world_size)

    lr = 1e-4
    warmup_steps = 1000

    if START_EPOCH > 1:
        trainer = create_trainer(rank, "./checkpoints/latent100000/encoder.pt", 
                                 "./checkpoints/motion%d/model.pt" % START_EPOCH)
        optimiser = torch.optim.AdamW(trainer.parameters(), lr=lr)
        optimiser.load_state_dict(torch.load("./checkpoints/motion%d/optimiser.pt" % START_EPOCH))
    else:
        trainer = create_trainer(rank, "./checkpoints/latent100000/encoder.pt")
        optimiser = torch.optim.AdamW(trainer.parameters(), lr=lr)

    dataloaders = get_dataloader(rank, world_size)

    for epoch in range(START_EPOCH + 1, EPOCHS + 1):
        for g in optimiser.param_groups:
            g["lr"] = lr * min((warmup_steps ** 0.5) / (epoch ** 0.5), epoch / warmup_steps)

        optimiser.zero_grad(set_to_none=True)

        # masked modelling
        with torch.no_grad():
            train_data = []
            for dataloader in dataloaders:
                train_data.append(next(iter(dataloader)).to(rank))
            train_data = torch.cat(train_data, dim=0)

        ret = trainer.train_masked_step(train_data)
        optimiser.step()

        print("Epoch %d - " % epoch, end="")
        print("I-E: %.4f; P-E: %.4f; T-E: %.4f; I-O: %.4f; P-O: %.4f; T-O: %.4f; " %
              (ret["loss_embeds_interp"], ret["loss_embeds_predict"], ret["loss_embeds_transition"],
               ret["loss_offset_interp"], ret["loss_offset_predict"], ret["loss_offset_transition"])
             )

        if epoch % 10 == 0:
            torch.cuda.empty_cache()

        if epoch % 5000 == 0 or epoch == 1:
            trainer.save_ckpt(epoch, optimiser)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("Running on %d GPUs" % world_size)
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
