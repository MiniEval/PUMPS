import colorsys
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from torch.amp import autocast, GradScaler

from data.skeleton import Skeleton
from models.PoseEncoder import PoseEncoder, PointCloudDecoder
from data.dataloader import MotionDataset

from torch_lap_cuda import solve_lap

    
EPOCHS = 100000
START_EPOCH = 0
BATCH_SIZE = 8 # per dataset
N_FRAMES = 128
N_POINTS = 256
MIN_STD = 0.025
MAX_STD = 0.075


class LatentTrainer:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.scaler = GradScaler()

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def reconstruct_loss(self, expected_points, actual_points, bgroups):
        batches = expected_points.shape[0]
        frames = expected_points.shape[1]
        n_bgroups = bgroups.shape[-1]

        points_e = []
        points_a_paired = []

        bgroups_idx = torch.argmax(bgroups, dim=-1)

        # with torch.no_grad():
        #     e_std, e_mean = torch.std_mean(expected_points, dim=2, keepdim=True)
        #     a_std, a_mean = torch.std_mean(actual_points, dim=2, keepdim=True)
        #     expected_points_norm = (expected_points - e_mean) / torch.clamp_min(e_std, 1e-4)
        #     actual_points_norm = (actual_points - a_mean) / torch.clamp_min(a_std, 1e-4)

        for g in range(n_bgroups):
            p_e = expected_points.transpose(0, 2)[bgroups_idx == g].transpose(0, 2)
            p_a = actual_points.transpose(0, 2)[bgroups_idx == g].transpose(0, 2)

            with torch.no_grad():
                p_e_ = torch.reshape(p_e.transpose(1, 2), (batches, p_e.shape[2], -1))
                p_a_ = torch.reshape(p_a.transpose(1, 2), (batches, p_a.shape[2], -1))

                dists = torch.cdist(p_e_, p_a_, p=2) ** 2
                indices = solve_lap(dists)
                indices = torch.broadcast_to(indices.view(batches, 1, p_e.shape[2], 1), (-1, frames, -1, 3))

            std, mean = torch.std_mean(p_e, dim=-2, keepdim=True)
            p_e = (p_e - mean) / std
            p_a = (p_a - mean) / std

            points_e.append(p_e)
            points_a_paired.append(torch.gather(p_a, 2, indices))

        points_e = torch.cat(points_e, dim=2)
        points_a_paired = torch.cat(points_a_paired, dim=2)
        dists = F.mse_loss(points_e, points_a_paired)

        return dists

    def train_ae_step(self, points, load_dist=None, alpha_dist=1.0):
        # points: [batch, frames, n_points, 3 + bgroups]
        # assume bgroups are same across batches

        batches = points.shape[0]
        frames = points.shape[1]
        n_points = points.shape[2]

        ret = dict()
        self.encoder.train()
        self.decoder.train()

        real_points = points.clone()
        bgroups = points[:, 0, :, 3:].clone()
        encoder_input = torch.reshape(points, (batches * frames, n_points, -1))
        
        with autocast(device_type="cuda"):
            embeds, offsets = self.encoder(encoder_input, return_offset=True)
            embeds = torch.reshape(embeds, (batches, frames, -1))
            offsets = torch.reshape(offsets, (batches, frames, 3))

            pred_points = self.decoder(embeds=embeds, offsets=offsets, bgroups=bgroups, apply_offset=True)
            dist_loss = self.reconstruct_loss(real_points[..., :3], pred_points[..., :3], bgroups[0])

            loss = dist_loss * alpha_dist

        self.scaler.scale(loss).backward()

        ret["real_points"] = real_points.clone().detach()
        ret["pred_points"] = pred_points.clone().detach()
        ret["dist_loss"] = dist_loss.item()

        return ret

    def save_ckpt(self, n_iter, optimiser=None):
        subfolder = "./checkpoints/latent%d/" % n_iter
        os.makedirs(subfolder, exist_ok=True)

        torch.save(self.encoder.module.state_dict(), subfolder + 'encoder.pt')
        torch.save(self.decoder.module.state_dict(), subfolder + 'decoder.pt')
        if optimiser:
            torch.save(optimiser.state_dict(), subfolder + 'optimiser.pt')
            torch.save(self.scaler.state_dict(),subfolder + 'scaler.pt')


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    
def create_trainer(rank):
    encoder = PoseEncoder(n_bgroups=5,
                        d_model=32,
                        k=8).to(rank)
    decoder = PointCloudDecoder(n_bgroups=5,
                                d_embed=512,
                                n_heads=8,
                                n_layers=16,
                                max_length=N_FRAMES,
                                dropout=0.2).to(rank)
    if START_EPOCH > 1:
        encoder.load_state_dict(torch.load("./checkpoints/latent%d/encoder.pt" % START_EPOCH))
        decoder.load_state_dict(torch.load("./checkpoints/latent%d/decoder.pt" % START_EPOCH))
        
    encoder = DDP(encoder, device_ids=[rank])
    decoder = DDP(decoder, device_ids=[rank])
    return LatentTrainer(encoder, decoder)


def get_dataloader(rank, world_size):
    dataset_h36m = MotionDataset("./datasets/processed/Human3.6M", sample_length=N_FRAMES, device=rank)
    dataset_style = MotionDataset("./datasets/processed/100STYLE", sample_length=N_FRAMES, device=rank)
    dataset_lafan = MotionDataset("./datasets/processed/LaFAN1", sample_length=N_FRAMES, device=rank)
    dataset_cmu = MotionDataset("./datasets/processed/CMU", sample_length=N_FRAMES, device=rank)
    dataset_amass = MotionDataset("./datasets/processed/AMASS", sample_length=N_FRAMES, device=rank)
    datasets = [dataset_h36m, dataset_lafan, dataset_style, dataset_cmu, dataset_amass]
    dataloaders = []
    for dataset in datasets:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloaders.append(DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                      collate_fn=dataset.collate_motion_points))
    return dataloaders


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    setup(rank, world_size)
    trainer = create_trainer(rank)
    dataloaders = get_dataloader(rank, world_size)
    
    lr = 1e-4
    warmup_steps = 1000
    optimiser = torch.optim.AdamW(trainer.parameters(), lr=lr)
    if START_EPOCH > 1:
        optimiser.load_state_dict(torch.load("./checkpoints/latent%d/optimiser.pt" % START_EPOCH))

    if VISUALISE:
        vis = Visualiser()
    log_file = open("latent_train_log.txt", "w")

    torch.cuda.empty_cache()

    for epoch in range(START_EPOCH + 1, EPOCHS + 1):
        for g in optimiser.param_groups:
            g["lr"] = lr * min((warmup_steps ** 0.5) / (epoch ** 0.5), epoch / warmup_steps)

        trainer.encoder.zero_grad()
        trainer.decoder.zero_grad()
        
        input_points = []
        for dataloader in dataloaders:
            input_points.append(next(iter(dataloader)).to(rank))

        p = torch.cat(input_points, dim=0)
        ret_ae = trainer.train_ae_step(p)
        trainer.scaler.step(optimiser)
        trainer.scaler.update()
        
        if dist.get_rank() == 0:
            dist_loss = ret_ae["dist_loss"]
            log = "Epoch %d - Dist: %.4f" % (epoch, dist_loss)
            log_file.write(log + "\n")
        
        if dist.get_rank() == 0 and (epoch % 10 == 0 or epoch == 1):
            print(log)
            log_file.flush()
            real_points = ret_ae["real_points"]
            pred_points = ret_ae["pred_points"]
            n_datasets = len(dataloaders)
            torch.cuda.empty_cache()

        if dist.get_rank() == 0 and (epoch % 10000 == 0 or epoch == 1):
            trainer.save_ckpt(epoch, optimiser)
            
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("Running on %d GPUs" % world_size)
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
