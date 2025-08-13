from glob import glob
import os
import numpy as np
# from pprint import pprint
from collections import OrderedDict
from argparse import ArgumentParser

def compute_metrics(eval_dir):
    metrics_dict = OrderedDict()

    # Pick Coke Can
    pick_horizon_num = len(glob(os.path.join(eval_dir, "google_pick_coke_can_1_v4", "**", "*switch*", "**", "*.mp4")))
    pick_horizon_succ_num = len(glob(os.path.join(eval_dir, "google_pick_coke_can_1_v4", "**", "*switch*", "**", "success*.mp4")))
    if pick_horizon_num == 100:
        pick_horizon_acc = pick_horizon_succ_num / 100
    else:
        print(f"pick_horizon_num: {pick_horizon_num}!!! {pick_horizon_succ_num / pick_horizon_num if pick_horizon_num>0 else np.nan}")
        pick_horizon_acc = np.nan
    metrics_dict['pick_horizon_acc'] = pick_horizon_acc

    pick_vertical_num = len(glob(os.path.join(eval_dir, "google_pick_coke_can_1_v4", "**", "*vertical*", "**", "*.mp4")))
    pick_vertical_succ_num = len(glob(os.path.join(eval_dir, "google_pick_coke_can_1_v4", "**", "*vertical*", "**", "success*.mp4")))
    if pick_vertical_num == 100:
        pick_vertical_acc = pick_vertical_succ_num / 100
    else:
        print(f"pick_vertical_num: {pick_vertical_num}!!! {pick_vertical_succ_num / pick_vertical_num if pick_vertical_num>0 else np.nan}")
        pick_vertical_acc = np.nan
    metrics_dict['pick_vertical_acc'] = pick_vertical_acc

    pick_stand_num = len(glob(os.path.join(eval_dir, "google_pick_coke_can_1_v4", "**", "*upright*", "**", "*.mp4")))
    pick_stand_succ_num = len(glob(os.path.join(eval_dir, "google_pick_coke_can_1_v4", "**", "*upright*", "**", "success*.mp4")))
    if pick_stand_num == 100:
        pick_stand_acc = pick_stand_succ_num / 100
    else:
        print(f"pick_stand_num: {pick_stand_num}!!! {pick_stand_succ_num / pick_stand_num if pick_stand_num>0 else np.nan}")
        pick_stand_acc = np.nan
    metrics_dict['pick_stand_acc'] = pick_stand_acc


    metrics_dict['avg_pick_acc'] = (pick_horizon_acc + pick_vertical_acc + pick_stand_acc) / 3



    # Move Near
    move_near_num = len(glob(os.path.join(eval_dir, "google_pick_coke_can_1_v4", "**", "Move*", "**", "*.mp4")))
    move_near_succ_num = len(glob(os.path.join(eval_dir, "google_pick_coke_can_1_v4", "**", "Move*", "**", "success*.mp4")))
    if move_near_num == 240:
        move_near_acc = move_near_succ_num / 240
    else:
        print(f"move_near_num: {move_near_num}!!! {move_near_succ_num / move_near_num if move_near_num>0 else np.nan}")
        move_near_acc = np.nan
    metrics_dict['move_near_acc'] = move_near_acc




    # Open / Close Drawer
    open_drawer_num = len(glob(os.path.join(eval_dir, "dummy_drawer", "**", "Open*", "**", "*.mp4")))
    open_drawer_succ_num = len(glob(os.path.join(eval_dir, "dummy_drawer", "**", "Open*", "**", "success*.mp4")))
    if open_drawer_num == 108:
        open_drawer_acc = open_drawer_succ_num / 108
    else:
        print(f"open_drawer_num: {open_drawer_num}!!! {open_drawer_succ_num / open_drawer_num if open_drawer_num>0 else np.nan}")
        open_drawer_acc = np.nan
    metrics_dict['open_drawer_acc'] = open_drawer_acc

    close_drawer_num = len(glob(os.path.join(eval_dir, "dummy_drawer", "**", "Close*", "**", "*.mp4")))
    close_drawer_succ_num = len(glob(os.path.join(eval_dir, "dummy_drawer", "**", "Close*", "**", "success*.mp4")))
    if close_drawer_num == 108:
        close_drawer_acc = close_drawer_succ_num / 108
    else:
        print(f"close_drawer_num: {close_drawer_num}!!! {close_drawer_succ_num / close_drawer_num if close_drawer_num>0 else np.nan}")
        close_drawer_acc = np.nan
    metrics_dict['close_drawer_acc'] = close_drawer_acc

    metrics_dict['avg_drawer_acc'] = (open_drawer_acc+close_drawer_acc)/2




    # Overall
    subtask_acc_list = []
    for k, v in metrics_dict.items():
        if not k.startswith("avg"):
            subtask_acc_list.append(v)

    avg_overall_acc_micro = np.mean(subtask_acc_list)
    metrics_dict['avg_overall_acc_micro'] = avg_overall_acc_micro


    print("*"*100)
    print(f"Evaluation results from {eval_dir}\n\n")
    for k, v in metrics_dict.items():
        print(f"{k}: {v}")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_dir", type=str)
    args = parser.parse_args()
    compute_metrics(args.eval_dir)
