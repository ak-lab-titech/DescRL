"""
元々あるLMDBデータセットからランダムにデータを取り出してより小さいデータセットを作成する
"""

import argparse
import time
import sys

import lmdb
import msgpack_numpy
import numpy as np

sys.path.append("/home/4/ud02274/navigation/myss")
sys.path.insert(0, "/home/4/ud02274/navigation/myss/sound-spaces")
sys.path.append("/home/4/ud02274/navigation/myss/habitat-lab")

from habitat.datasets import make_dataset
from ss_baselines.savi.config.default import get_config

def main(
    config,
    source_dataset_path,
    target_dataset_path,
    source_dataset_size,
    target_dataset_size,
):
    dataset = make_dataset(
        id_dataset=config.DATASET.TYPE,
        config=config.DATASET,
    )
    print(f"dataset.split: {config.DATASET.SPLIT}")
    episodes = dataset.episodes

    env_source = lmdb.open(source_dataset_path, readonly=True, lock=False)
    env_target = lmdb.open(target_dataset_path, map_size=int(5 * 1.1e12))

    indices = np.random.choice(source_dataset_size, size=target_dataset_size, replace=False)
    
    s = time.time()
    for i, index in enumerate(indices):
        with env_source.begin() as txn:
            value = txn.get(str(index).encode('latin-1'))
            value = msgpack_numpy.unpackb(value, object_hook=msgpack_numpy.decode)
        
        with env_target.begin(write=True) as txn:
            step = value[2][-1, 0, 3]
            instruction = np.array(episodes[index].instructions)[:, int(step)]
            value[6] = instruction
            txn.put(
                f"{i}".encode(), 
                msgpack_numpy.packb(
                    value, use_bin_type=True
                ),
            )
        if i % 100 == 0:
            print(f"i={i}: {time.time() - s}")
            s = time.time()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str) # DATASET.SPLITを用変更
    parser.add_argument('--source-dataset-path', type=str)
    parser.add_argument('--source-data-num', type=int)
    parser.add_argument('--data-num', type=int)

    args = parser.parse_args()
    
    config = get_config(args.config)

    args = parser.parse_args()
    main(
        config=config.TASK_CONFIG,
        source_dataset_path=args.source_dataset_path,
        target_dataset_path=f"{args.source_dataset_path}_{args.data_num}",
        source_dataset_size=args.source_data_num,
        target_dataset_size=args.data_num,
    )
