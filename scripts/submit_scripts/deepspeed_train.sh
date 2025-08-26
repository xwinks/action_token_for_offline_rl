
export MASTER_PORT=29999
export PROJECT_ROOT=.
CONFIG_NAME=$1
cd ${PROJECT_ROOT}/latent_motion_tokenizer
deepspeed --num_gpus=2 train/lerobot_train_latent_motion_tokenizer.py --config_path "${PROJECT_ROOT}/configs/train/${CONFIG_NAME}.yaml"
