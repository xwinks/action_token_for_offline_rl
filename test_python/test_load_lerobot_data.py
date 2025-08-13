from pprint import pprint

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    repo_id = "test"
    root = "/data_16T/lerobot_openx/cmu_stretch_lerobot"
    camera_key = "observation.images.image"

    delta_timestamps = {
        # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
        camera_key: [-2, -1, 0],
        # loads 6 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
        "observation.state": [-2, -1, 0],
        # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
        "action": [t / 5 for t in range(50)],
    }

    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    ds = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
    data_0 = ds[0]
    print(data_0['observation.images.image'])
    print(data_0['action'].shape)
    print(data_0['observation.state'].shape)

    dataloader = torch.utils.data.DataLoader(
        ds,
        num_workers=0,
        batch_size=32,
        shuffle=True,
    )

    for batch in dataloader:
        print("the batch keys: ", batch.keys())
        print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
        print(f"{batch['observation.state'].shape=}")  # (32, 6, c)
        print(f"{batch['action'].shape=}")  # (32, 64, c)
        break