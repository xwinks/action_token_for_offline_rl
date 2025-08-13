import numpy as np
import pandas as pd
import json
import os
if __name__ == "__main__":
    
    pesudo_label_path = "debug_pesudo_label/evaluation_results.json"
    data_dir = "/home/v-wenhuitan/wenke_workspace/wenke_data/workspace/action_token_for_offline_rl/lerobot_openx/xwk_debug_cmu_stretch_lerobot/data/chunk-000"
    
    with open(pesudo_label_path, "r") as f:
        pesudo_label = json.load(f)
        
    # print(pesudo_label)
    episode_id = 0
    single_traj_labels = []
    for single_annotation in pesudo_label:
        annotation_episode_id = single_annotation["episode_id"]
        # print("the annotation_episode_id is: ", annotation_episode_id)
        
        if annotation_episode_id == episode_id:
            frame_id = single_annotation["frame_id"]
            gt_latent_motion_ids = single_annotation["gt_latent_motion_ids"]
            # print("the gt_latent_motion_ids is: ", gt_latent_motion_ids)
            single_traj_labels.append(gt_latent_motion_ids)
        else:
            episode_data = pd.read_parquet(os.path.join(data_dir, f"episode_{episode_id:06d}.parquet"))
            print("the length of the episode_data is: ", len(single_traj_labels))
            episode_data['generated_latent_motion_ids'] = single_traj_labels
            episode_data.to_parquet(os.path.join(data_dir, f"episode_{episode_id:06d}.parquet"))
            episode_id += 1
            single_traj_labels = []
            frame_id = single_annotation["frame_id"]
            gt_latent_motion_ids = single_annotation["gt_latent_motion_ids"]
            single_traj_labels.append(gt_latent_motion_ids)
    
    episode_data = pd.read_parquet(os.path.join(data_dir, f"episode_{episode_id:06d}.parquet"))
    print("the length of the episode_data is: ", len(single_traj_labels))
    episode_data['generated_latent_motion_ids'] = single_traj_labels
    episode_data.to_parquet(os.path.join(data_dir, f"episode_{episode_id:06d}.parquet"))
    
        # episode_data = pd.read_json(os.path.join(data_dir, f"{episode_id}.json"))
        # print(episode_data)
    
    
    