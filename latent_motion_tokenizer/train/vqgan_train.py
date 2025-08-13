import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import argparse
import json
from torch.utils.data import DataLoader
import omegaconf
import hydra
from functools import partial
from transformers import AutoTokenizer
from common.models.model_utils import load_model
from common.processors.preprocessor_utils import get_rgb_preprocessor
from latent_motion_tokenizer.src.trainers.latent_motion_tokenizer_trainer import LatentMotionTokenizer_Trainer
from torch.utils.data import DataLoader
from functools import partial
from common.data.data_utils import load_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


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
    repo_id = "test"
    root = "/data_16T/lerobot_openx/cmu_stretch_lerobot"
    camera_key = "observation.images.image"
    delta_timestamps = {
        # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
        camera_key: [-1, 0],
        # loads 6 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
        "observation.state": [-1, 0],
        # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
        "action": [-1,0],
    }
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
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        obs_name = camera_key,
        **cfg['training_config']
    )

    # Start Training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/group/40101/milkcychen/Moto/latent_motion_tokenizer/configs/train/data_calvin-vq_size128_dim32_num8_legacyTrue-vision_MaeLarge-decoder_queryFusionModeAdd_Patch196_useMaskFalse-mformer_legacyTrue-train_lr0.0001_bs256-aug_shiftTrue_resizedCropFalse.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    print(f"the cfg is: {cfg}")
    main(cfg)

    


