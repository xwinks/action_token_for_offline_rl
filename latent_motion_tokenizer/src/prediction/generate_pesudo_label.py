import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import os
from time import time
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
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
import json
class LatentMotionTokenizer_Evaluator:
    def __init__(
        self,
        latent_motion_tokenizer,
        rgb_preprocessor,
        eval_dataloader,
        resume_ckpt_path=None,
        obs_name="rgb_initial",
    ):
        if resume_ckpt_path is not None:
            print(f"resuming Latent Motion Tokenizer from {resume_ckpt_path} ...")
            missing_keys, unexpected_keys = latent_motion_tokenizer.load_state_dict(torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)
        
        self.obs_name = obs_name

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.latent_motion_tokenizer = latent_motion_tokenizer.to(device)
        self.eval_prefetcher = DataPrefetcher(eval_dataloader, device)
        self.rgb_preprocessor = rgb_preprocessor.to(device)


    @torch.no_grad()
    def eval_latent_motion_reconstruction(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        evaluation_results = []
        
        print(f"Saving visualization results to {save_dir} ...")
        # batch, _ = self.eval_prefetcher.next_without_none()

        while True:
            batch_tuple = self.eval_prefetcher.next()
            batch, _ = batch_tuple
            if batch is None:
                break
            
            pre_frame = batch[self.obs_name][:, 0]
            pre_frame = pre_frame.unsqueeze(1)
            cur_frame = batch[self.obs_name][:, 1]
            cur_frame = cur_frame.unsqueeze(1)
            
            single_image = pre_frame[0][0]
            # Convert tensor to numpy and ensure proper format for cv2.imwrite
            single_image_np = single_image.cpu().numpy()
            # Ensure image is in the correct format (H, W, C) with proper channel order
            if len(single_image_np.shape) == 3 and single_image_np.shape[0] in [1, 3, 4]:
                # If channels are in first dimension, transpose to (H, W, C)
                single_image_np = single_image_np.transpose(1, 2, 0)
            # Ensure values are in proper range [0, 255] for uint8
            if single_image_np.max() <= 1.0:
                single_image_np = (single_image_np * 255).astype('uint8')
            else:
                single_image_np = single_image_np.astype('uint8')

            
            
            orig_rgb_seq = torch.cat([pre_frame, cur_frame], dim=1) # (b, 2, c, h, w)

            
            rgb_seq = self.rgb_preprocessor(orig_rgb_seq, train=True)

            self.latent_motion_tokenizer.eval()
            motion_ids = self.latent_motion_tokenizer(
                cond_pixel_values=rgb_seq[:,0],
                target_pixel_values=rgb_seq[:,1],
                return_recons_only=False,
                return_motion_token_ids_only = True
            )
            # import pdb; pdb.set_trace()

            gt_latent_motion_ids = motion_ids.detach().cpu() # (b, per_latent_motion_len)
            # import pdb; pdb.set_trace()
            print("the gt_latent_motion_ids is: ", gt_latent_motion_ids.shape)
            
            frame_ids = batch['frame_index']
            episode_ids = batch['episode_index']
            task_ids = batch['task_index']
            
            for single_idx in range(len(frame_ids)):
                evaluation_results.append({
                    'frame_id': frame_ids[single_idx].item(),
                    'episode_id': episode_ids[single_idx].item(),
                    'task_id': task_ids[single_idx].item(),
                    'gt_latent_motion_ids': gt_latent_motion_ids[single_idx].tolist()
                })
            
            print("the episode_ids is: ", episode_ids[0])
            
            print("the evaluation_results is: ", evaluation_results)
            
            with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(evaluation_results, f)
                
            print(f"Evaluation results saved to {save_dir} ...")
                
            