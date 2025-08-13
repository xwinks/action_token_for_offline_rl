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

from latent_motion_tokenizer.src.prediction.generate_pesudo_label import LatentMotionTokenizer_Evaluator

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
    eval_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
    )
    
    print("the batch size is: ", cfg['dataloader_config']['bs_per_gpu'])
    
    dataloader_cls = partial(
        DataLoader, 
        pin_memory=True, # Accelerate data reading
        shuffle=False,
        persistent_workers=False,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
        prefetch_factor= cfg['dataloader_config']['prefetch_factor']
    )
    eval_dataloader = dataloader_cls(eval_dataset)
    

    evaluator = LatentMotionTokenizer_Evaluator(
        latent_motion_tokenizer=latent_motion_tokenizer,
        rgb_preprocessor=rgb_preprocessor,
        eval_dataloader=eval_dataloader,
        resume_ckpt_path=cfg['resume_ckpt_path'],
        obs_name=camera_key
    )

    evaluator.eval_latent_motion_reconstruction(save_dir=cfg['save_dir'])
    # Start Training
    # trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/group/40101/milkcychen/Moto/latent_motion_tokenizer/configs/train/data_calvin-vq_size128_dim32_num8_legacyTrue-vision_MaeLarge-decoder_queryFusionModeAdd_Patch196_useMaskFalse-mformer_legacyTrue-train_lr0.0001_bs256-aug_shiftTrue_resizedCropFalse.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    print(f"the cfg is: {cfg}")
    main(cfg)

    


