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
from torch.utils.data import DataLoader
from functools import partial
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import yaml
from action_prediction.src.models.motogpt_for_action import MotoGPTActionPrediction
from action_prediction.src.trainers.action_prediction_trainer import ActionPredictionTrainer

def main(cfg):
    # Prepare rgb_processor
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    action_prediction_transformer_config_path = cfg['action_prediction_transformer_config_path']
    print(f"initializing action prediction transformer from {action_prediction_transformer_config_path} ...")
    action_prediction_transformer_config = omegaconf.OmegaConf.load(action_prediction_transformer_config_path)
    action_prediction_transformer = hydra.utils.instantiate(action_prediction_transformer_config)
    action_prediction_transformer.config = action_prediction_transformer_config     

    print("the action prediction transformer is: ", action_prediction_transformer)

    # Preprepare Dataloaders
    dataset_config_path = cfg['dataset_config_path']
    extra_data_config = {
        'sequence_length': 1,
        'do_extract_future_frames': True,
        'do_extract_past_frames': True,
    }

    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    repo_id = dataset_config['repo_id']
    root = dataset_config['root']
    camera_key = dataset_config['camera_key']
    delta_timestamps = dataset_config['delta_timestamps']
    
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    print(f"Total number of episodes: {ds_meta.total_episodes}")
    print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
    print(f"Frames per second used during data collection: {ds_meta.fps}")
    print(f"Robot type: {ds_meta.robot_type}")
    print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")
    print("Tasks:")
    
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
    lang_tokenizer_config = cfg['lang_tokenizer']
    lang_tokenizer = hydra.utils.instantiate(lang_tokenizer_config)
    
    trainer = ActionPredictionTrainer(
        action_prediction_transformer=action_prediction_transformer,
        rgb_preprocessor=rgb_preprocessor,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        **cfg['training_config'],
        lang_tokenizer=lang_tokenizer
    )
    trainer.train()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/train/train.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    print(f"the cfg is: {cfg}")
    main(cfg)
    
    