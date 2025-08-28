import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import os
from time import time
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import torch
from common.data.datasets import DataPrefetcher
from latent_motion_tokenizer.src.trainers.trainer_utils import visualize_latent_motion_reconstruction
import omegaconf
from glob import glob
import shutil
from collections import defaultdict
from latent_motion_tokenizer.src.trainers.optimizer import get_optimizer, LinearWarmup_CosineAnnealing
from contextlib import contextmanager
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import deepspeed
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
class LatentMotionTokenizer_Trainer:
    def __init__(
        self,
        latent_motion_tokenizer,
        rgb_preprocessor,
        ds_meta,
        train_dataloader,
        eval_dataloader,
        save_path,
        save_epochs=1,
        save_steps=10000,
        num_epochs=20,
        print_steps=100,
        lr_max=0.0001,
        weight_decay=0.,
        num_warmup_epochs=1,
        gradient_accumulation_steps=4,
        resume_ckpt_path=None,
        bs_per_gpu=32,
        max_epoch=None,
        obs_name="rgb_initial",
        use_deepspeed=False,
        deepspeed_config_path=None,
        
    ):
        if resume_ckpt_path is not None:
            print(f"resuming Latent Motion Tokenizer from {resume_ckpt_path} ...")
            missing_keys, unexpected_keys = latent_motion_tokenizer.load_state_dict(torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)
        
        self.obs_name = obs_name
        self.ds_meta = ds_meta
        if deepspeed_config_path is None:
            raise ValueError("deepspeed_config_path must be provided for DeepSpeed training")
        # init distributed
        if not dist.is_initialized():
            deepspeed.init_distributed()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(self.local_rank)
        
        print("self.local_rank: ", self.local_rank)
        print("self.rank: ", self.rank)
        print("self.world_size: ", self.world_size)
        self._device = torch.device('cuda', self.local_rank)

        total_prints_per_epoch = len(train_dataloader.dataset) // (print_steps * bs_per_gpu * self.world_size)

        base_optimizer = get_optimizer(
                        [p for n, p in latent_motion_tokenizer.named_parameters() if p.requires_grad], 
                        lr=lr_max, 
                        wd=weight_decay
                    )
        
        # DeepSpeed engine init
        self.engine, self.optimizer, _, _ = deepspeed.initialize(
            model=latent_motion_tokenizer,
            optimizer=base_optimizer,
            model_parameters=[p for n, p in latent_motion_tokenizer.named_parameters() if p.requires_grad],
            config=deepspeed_config_path
        )

        linear_warmup_total_iters = min(num_warmup_epochs*total_prints_per_epoch, 5000000 // (print_steps * bs_per_gpu * self.world_size))
        self.scheduler = LinearWarmup_CosineAnnealing(
                        optimizer=base_optimizer,
                        linear_warmup_start_factor=0.5,
                        linear_warmup_total_iters=linear_warmup_total_iters,
                        cosine_annealing_T_max=num_epochs*total_prints_per_epoch-linear_warmup_total_iters,
                        cosine_annealing_eta_min=5e-5
                    )
        # ensure engine optimizer lr matches scheduler at start
        self._apply_current_lr_to_engine()

        # Build distributed dataloaders
        self.train_loader = self._build_distributed_loader(train_dataloader, shuffle=True)
        self.eval_loader = self._build_distributed_loader(eval_dataloader, shuffle=False)
        
        self.writer = SummaryWriter(os.path.join(save_path, 'logs')) if self.is_main else None
        self.total_prints_per_epoch = total_prints_per_epoch
        self.latent_motion_tokenizer = self.engine
        self.train_prefetcher = DataPrefetcher(self.train_loader, self.device)
        self.eval_prefetcher = DataPrefetcher(self.eval_loader, self.device)
        self.rgb_preprocessor = rgb_preprocessor.to(self.device)
        self.save_path = save_path
        self.save_epochs = save_epochs
        self.save_steps = save_steps
        self.max_epoch = max_epoch
        self.num_epochs = num_epochs
        self.print_steps = print_steps
        self.bs_per_gpu = bs_per_gpu

    def _build_distributed_loader(self, loader, shuffle):
        if self.world_size <= 1:
            return loader
        sampler = DistributedSampler(loader.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle, drop_last=False)
        kwargs = dict(
            batch_size=loader.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
            collate_fn=loader.collate_fn,
            persistent_workers=getattr(loader, 'persistent_workers', False),
        )
        prefetch_factor = getattr(loader, 'prefetch_factor', None)
        if loader.num_workers > 0 and prefetch_factor is not None:
            kwargs['prefetch_factor'] = prefetch_factor
        return DataLoader(loader.dataset, **kwargs)

    def _apply_current_lr_to_engine(self):
        last_lrs = self.scheduler.get_last_lr()
        for i, group in enumerate(self.engine.optimizer.param_groups):
            group['lr'] = last_lrs[min(i, len(last_lrs)-1)]

    @property
    def device(self):
        return self._device

    @property
    def is_main(self):
        return (not dist.is_initialized()) or (self.rank == 0)

    @property
    def process_index(self):
        return self.rank

    def print(self, msg):
        if self.is_main:
            print(msg)

    def save_checkpoint(self, save_dir):
        unwrapped_latent_motion_tokenizer = self.engine.module
        state_dict = unwrapped_latent_motion_tokenizer.get_state_dict_to_save()
        
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        omegaconf.OmegaConf.save(unwrapped_latent_motion_tokenizer.config, os.path.join(save_dir, "config.yaml"))

        self.print(f"A new model checkpoint is saved to {save_dir}!!!")
        
    def train(self):
        step = 0
        
        for epoch in range(self.num_epochs+1):
            # set epoch for distributed sampler to reshuffle
            if isinstance(getattr(self.train_loader, 'sampler', None), DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            if epoch != 0:
                if dist.is_initialized():
                    dist.barrier()
                save_dir = os.path.join(self.save_path, f'saved_epoch_{epoch}_step_{step}')

                if self.is_main:
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_checkpoint(save_dir)
                    
                visualization_dir = os.path.join(save_dir, 'visualization')
                self.eval_latent_motion_reconstruction(visualization_dir)

                if epoch == self.num_epochs:
                    break
                if (self.max_epoch is not None) and (epoch >= self.max_epoch):
                    break

            log_loss = {}
            eval_log_loss = {}
            
            cum_load_time = 0 
            clock = time()
            batch_idx = 0
            batch, load_time = self.train_prefetcher.next()
            
            # 计算当前epoch的总批次数
            total_batches = self.ds_meta.total_frames  // self.bs_per_gpu // self.world_size
            temp_batch = batch
            temp_load_time = load_time
            # while temp_batch is not None:
            #     total_batches += 1
            #     temp_batch, temp_load_time = self.train_prefetcher.next()
            
            # 重新初始化prefetcher
            self.train_prefetcher = DataPrefetcher(self.train_loader, self.device)
            batch, load_time = self.train_prefetcher.next()
            
            # 创建进度条
            if self.is_main:
                pbar = tqdm(
                    total=total_batches,
                    desc=f'Epoch {epoch}/{self.num_epochs}',
                    leave=True,
                    position=0,
                    ncols=120
                )
            
            while batch is not None:
                self.engine.train()
                self.engine.zero_grad()
                loss = self.calculate_loss(batch, train=True)
                self.engine.backward(loss['loss'])
                self.engine.step()

                for key in loss:
                    if key not in log_loss:
                        log_loss[key] = torch.tensor(0.0, device=self.device)
                    log_loss[key] = log_loss[key] + loss[key].detach() / self.print_steps

                cum_load_time += load_time / self.print_steps

                if (batch_idx+1) % self.print_steps == 0:

                    with torch.no_grad():
                        batch, _ = self.eval_prefetcher.next_without_none()
                        self.engine.eval()
                        loss = self.calculate_loss(batch, train=True)
                        for key in loss:
                            eval_log_loss[key] = loss[key].detach()

                    self.log(log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step)
                    
                    if self.is_main:
                        loss_str = ' '.join([f'{k}: {v:.4f}' for k, v in log_loss.items()])
                        eval_loss_str = ' '.join([f'eval_{k}: {v:.4f}' for k, v in eval_log_loss.items()])
                        pbar.set_postfix({
                            'loss': loss_str,
                            'eval_loss': eval_loss_str,
                            'lr': f'{self.scheduler.get_last_lr()[0]:.6f}',
                            'step': step
                        })
                    
                    log_loss = {}
                    eval_log_loss = {}

                    cum_load_time = 0
                    clock = time()
                    self.scheduler.step()
                    self._apply_current_lr_to_engine()

                if step % self.save_steps == 0:
                    if dist.is_initialized():
                        dist.barrier()
                    save_dir = os.path.join(self.save_path, f'temp_epoch_{epoch}_step_{step}')
                    if self.is_main:
                        existing_ckpt_dirs = glob(os.path.join(self.save_path, f'temp_epoch_*_step_*'))
                        for existing_ckpt_dir in existing_ckpt_dirs:
                            if existing_ckpt_dir != save_dir:
                                shutil.rmtree(existing_ckpt_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        self.save_checkpoint(save_dir)

                    visualization_dir = os.path.join(save_dir, 'visualization')
                    self.eval_latent_motion_reconstruction(visualization_dir)

                batch_idx += 1
                step += 1
                batch, load_time = self.train_prefetcher.next()
                
                if self.is_main:
                    pbar.update(1)
            
            if self.is_main:
                pbar.close()



    @torch.no_grad()
    def eval_latent_motion_reconstruction(self, visualization_dir):
        os.makedirs(visualization_dir, exist_ok=True)
        self.print(f"Saving visualization results to {visualization_dir} ...")
        batch, _ = self.eval_prefetcher.next_without_none()

        pre_frame = batch[self.obs_name][:, 0]
        pre_frame = pre_frame.unsqueeze(1)
        cur_frame = batch[self.obs_name][:, 1]
        cur_frame = cur_frame.unsqueeze(1)
        
        single_image = pre_frame[0][0]
        single_image_np = single_image.cpu().numpy()
        if len(single_image_np.shape) == 3 and single_image_np.shape[0] in [1, 3, 4]:
            single_image_np = single_image_np.transpose(1, 2, 0)
        if single_image_np.max() <= 1.0:
            single_image_np = (single_image_np * 255).astype('uint8')
        else:
            single_image_np = single_image_np.astype('uint8')
        cv2.imwrite(os.path.join(visualization_dir, "single_image.png"), single_image_np)
        
        orig_rgb_seq = torch.cat([pre_frame, cur_frame], dim=1)
        
        rgb_seq = self.rgb_preprocessor(orig_rgb_seq, train=True)
        model_dtype = next(self.engine.module.parameters()).dtype
        rgb_seq = rgb_seq.to(model_dtype)

        self.engine.eval()
        outputs = self.engine(
            cond_pixel_values=rgb_seq[:,0],
            target_pixel_values=rgb_seq[:,1],
            return_recons_only=True
        )
            
        recons_rgb_future = self.rgb_preprocessor.post_process(outputs["recons_pixel_values"]).detach().cpu()
        gt_latent_motion_ids = outputs["indices"].detach().cpu()
        orig_rgb_seq = self.rgb_preprocessor.post_process(rgb_seq).detach().cpu()

        post_process_rgb_origin = orig_rgb_seq[0][0]
        single_image_np = post_process_rgb_origin.cpu().numpy()
        if len(single_image_np.shape) == 3 and single_image_np.shape[0] in [1, 3, 4]:
            single_image_np = single_image_np.transpose(1, 2, 0)
        if single_image_np.max() <= 1.0:
            single_image_np = (single_image_np * 255).astype('uint8')
        else:
            single_image_np = single_image_np.astype('uint8')
        cv2.imwrite(os.path.join(visualization_dir, "post_process_rgb_origin.png"), single_image_np)

        for i in range(orig_rgb_seq.shape[0]):
            visualize_latent_motion_reconstruction(
                initial_frame=orig_rgb_seq[i,0],
                next_frame=orig_rgb_seq[i,1],
                recons_next_frame=recons_rgb_future[i],
                latent_motion_ids=gt_latent_motion_ids[i],
                path=os.path.join(visualization_dir, f"{self.process_index}-{i}.png")
            )


    def calculate_loss(self, batch, train):
        pre_frame = batch[self.obs_name][:, 0]
        pre_frame = pre_frame.unsqueeze(1)
        cur_frame = batch[self.obs_name][:, 1]
        cur_frame = cur_frame.unsqueeze(1)
        rgb_seq = torch.cat([pre_frame, cur_frame], dim=1)
        rgb_seq = self.rgb_preprocessor(rgb_seq, train=train)
        model_dtype = next(self.engine.module.parameters()).dtype
        rgb_seq = rgb_seq.to(model_dtype)

        loss = self.engine(
            cond_pixel_values=rgb_seq[:,0],
            target_pixel_values=rgb_seq[:,1]
        )
        # print("the loss is: ", loss)

        return loss


    def _reduce_mean(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / self.world_size
        return tensor

    def log(self, log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step):
        for key in log_loss:
            log_loss[key] = self._reduce_mean(log_loss[key])
        for key in eval_log_loss:
            eval_log_loss[key] = self._reduce_mean(eval_log_loss[key])
        load_pecnt = torch.tensor(cum_load_time / (time()-clock), device=self.device)
        load_pecnt = self._reduce_mean(load_pecnt)
        fps = (self.bs_per_gpu*self.print_steps*2) / (time()-clock)
        fps = self._reduce_mean(torch.tensor(fps, device=self.device)) * self.world_size

        text = 'Train Epoch: {} [{}/{} ({:.0f}%)] FPS:{:.5f} Load Pertentage:{:.5f} LR:{}'.format(
            epoch, 
            batch_idx * self.bs_per_gpu * self.world_size, 
            len(self.train_prefetcher), 
            100. * batch_idx * self.bs_per_gpu * self.world_size / len(self.train_prefetcher),
            fps,
            load_pecnt,
            self.scheduler.get_last_lr()[0],
        )
        for key in log_loss:
            text = text + ' {}: {:.5f}'.format(key, log_loss[key])
        for key in eval_log_loss:
            text = text + ' eval_{}: {:.5f}'.format(key, eval_log_loss[key])
        self.print(text)
        if self.is_main and self.writer is not None:
            for key in log_loss:
                self.writer.add_scalar(key, log_loss[key].item() if isinstance(log_loss[key], torch.Tensor) else log_loss[key], step)
            for key in eval_log_loss:
                self.writer.add_scalar('eval_'+key, eval_log_loss[key].item() if isinstance(eval_log_loss[key], torch.Tensor) else eval_log_loss[key], step)
            self.writer.add_scalar("learning rate", self.scheduler.get_last_lr()[0], step)
            self.writer.add_scalar("FPS", fps.item() if isinstance(fps, torch.Tensor) else fps, step)
            self.writer.add_scalar("loading time in total time", load_pecnt.item() if isinstance(load_pecnt, torch.Tensor) else load_pecnt, step)
