import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import argparse
import json
import os
from torch.utils.data import DataLoader
import omegaconf
import hydra
from functools import partial
from transformers import AutoTokenizer
from common.models.model_utils import load_model
from common.processors.preprocessor_utils import get_rgb_preprocessor
from latent_motion_tokenizer.src.trainers.deepspeed_latent_motion_tokenizer_trainer import LatentMotionTokenizer_Trainer
from torch.utils.data import DataLoader
from functools import partial
from common.data.data_utils import load_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import yaml

def main(cfg):
    # Prepare Latent Motion Tokenizer
    latent_motion_tokenizer_config_path = cfg['latent_motion_tokenizer_config_path']
    print(f"initializing Latent Motion Tokenizer from {latent_motion_tokenizer_config_path} ...")
    latent_motion_tokenizer_config = omegaconf.OmegaConf.load(latent_motion_tokenizer_config_path)
    latent_motion_tokenizer = hydra.utils.instantiate(latent_motion_tokenizer_config)
    latent_motion_tokenizer.config = latent_motion_tokenizer_config

    # Prepare rgb_processor
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    # Preprepare Dataloaders
    dataset_config_path = cfg['dataset_config_path']
    extra_data_config = {
        'sequence_length': 1,
        'do_extract_future_frames': True,
        'do_extract_action': False
    }
    
    # train_dataset, eval_dataset = load_dataset(dataset_config_path, extra_data_config)
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    repo_id = dataset_config['repo_id']
    root = dataset_config['root']
    camera_key = dataset_config['camera_key']
    delta_timestamps = dataset_config['delta_timestamps']
    train_ratio = 0.9
    
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    print(f"Total number of episodes: {ds_meta.total_episodes}")
    print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
    print(f"Frames per second used during data collection: {ds_meta.fps}")
    print(f"Robot type: {ds_meta.robot_type}")
    print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")
    print("Tasks:")
    train_episodes_list = list(range(int(len(ds_meta.episodes) * train_ratio)))
    eval_episodes_list = list(range(int(len(ds_meta.episodes) * train_ratio), len(ds_meta.episodes)))
    train_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        # episodes=train_episodes_list,
        delta_timestamps=delta_timestamps,
        # image_transforms=rgb_preprocessor
    )
    eval_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        # episodes=eval_episodes_list,
        delta_timestamps=delta_timestamps,
        # image_transforms=rgb_preprocessor
    )
    
    print("the batch size is: ", cfg['dataloader_config']['bs_per_gpu'])
    
    dataloader_cls = partial(
        DataLoader, 
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        persistent_workers=True,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
        prefetch_factor= cfg['dataloader_config']['prefetch_factor']
    )
    train_dataloader = dataloader_cls(train_dataset)
    eval_dataloader = dataloader_cls(eval_dataset)
    
    # Prepare Trainer
    trainer = LatentMotionTokenizer_Trainer(
        latent_motion_tokenizer=latent_motion_tokenizer,
        rgb_preprocessor=rgb_preprocessor,
        ds_meta = ds_meta,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        obs_name = camera_key,
        use_deepspeed=True,
        deepspeed_config_path=cfg['deepspeed_config_path'],
        **cfg['training_config']
    )

    # Start Training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/group/40101/milkcychen/Moto/latent_motion_tokenizer/configs/train/data_calvin-vq_size128_dim32_num8_legacyTrue-vision_MaeLarge-decoder_queryFusionModeAdd_Patch196_useMaskFalse-mformer_legacyTrue-train_lr0.0001_bs256-aug_shiftTrue_resizedCropFalse.yaml")
    parser.add_argument('--local_rank', type=int, default=-1)
    args, _ = parser.parse_known_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    print(f"the cfg is: {cfg}")
    main(cfg)

    


