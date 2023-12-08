import argparse
import logging
import os
import sys

os.environ['MAGNUM_LOG'] = "quiet"
os.environ['HABITAT_SIM_LOG'] = "quiet"

sys.path.insert(0, "/home/0/19B30511/av-nav/myss/sound-spaces")
sys.path.append("/home/0/19B30511/av-nav/myss/habitat-lab")


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import torch

from ss_baselines.savi.config.default import get_config
from ss_baselines.savi.iprl_pretraining.iprl_pretraining_trainer import IPRLPretrainingTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        # required=True,
        default='train',
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--model-dir",
        default='data/models/output',
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    # repo = git.Repo(search_parent_directories=True)
    # logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir, exist_ok=True)

    # run exp
    config = get_config(args.config, args.opts, args.model_dir, 'train', False)
    trainer = IPRLPretrainingTrainer(config)
    torch.set_num_threads(1)

    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().setLevel(level)

    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval":
        trainer.eval(1, -1, config.USE_LAST_CKPT)


if __name__ == "__main__":
    f = open("debug.txt", "w")
    f.write(f"RUN Pretraining IPRL\n")
    f.write(f"cuda is available in run.py: {torch.cuda.is_available()}\n")
    f.close()
    main()
