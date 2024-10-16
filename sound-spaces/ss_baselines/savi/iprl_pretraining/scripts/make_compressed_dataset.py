import os
import gzip

import json
import numpy as np


def main(dataset_dir_path, compression_ratio):
    compressed_dataset_dir_path = dataset_dir_path + f"_cr_{compression_ratio}"
    os.makedirs(f"{compressed_dataset_dir_path}/content", exist_ok=True)

    scene_file_names = [
        f for f in os.listdir(f"{dataset_dir_path}/content") if os.path.isfile(os.path.join(f"{dataset_dir_path}/content", f))
    ]
    for file_name in scene_file_names:
        with gzip.open(f"{dataset_dir_path}/content/{file_name}", "rb") as file:
            data = json.load(file)

        compressed_data = {
            "scene": data["scene"],
            "episodes": np.random.choice(data["episodes"], size=int(len(data["episodes"])*compression_ratio), replace=False).tolist(),
        }
        print(compressed_data["scene"], len(compressed_data["episodes"]), "/", len(data["episodes"]))

        compressed_data = json.dumps(compressed_data)
        with gzip.open(f"{compressed_dataset_dir_path}/content/{file_name}", "wt") as f:
            f.write(compressed_data)


if __name__=="__main__":
    main(
        dataset_dir_path=f"./data/datasets/semantic_audionav/mp3d/v1/train",
        compression_ratio=0.2,
    )
