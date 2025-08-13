export CUDA_VISIBLE_DEVICES=1

export PROJECT_ROOT=/home/v-wenhuitan/wenke_workspace/wenke_data/workspace/action_token_for_offline_rl
export CONFIG_NAME="train_action_prediction"

cd ${PROJECT_ROOT}/action_prediction
accelerate launch --main_process_port 29501 train/train_action_prediction.py --config_path "${PROJECT_ROOT}/action_prediction/configs/train/${CONFIG_NAME}.yaml"

