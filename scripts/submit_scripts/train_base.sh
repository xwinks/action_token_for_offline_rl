
export PROJECT_ROOT=.
CONFIG_NAME=$1
cd ${PROJECT_ROOT}/latent_motion_tokenizer
accelerate launch --main_process_port 29501 train/lerobot_train_latent_motion_tokenizer.py --config_path "${PROJECT_ROOT}/configs/train/${CONFIG_NAME}.yaml"