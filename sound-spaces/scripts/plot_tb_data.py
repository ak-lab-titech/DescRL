import os
from typing import List
import datetime
import argparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import japanize_matplotlib
import yaml


def get_tb_data_dict(model_name: str):
    dir_path = f'./data/models/ss2/replica/{model_name}/tb'
    
    if len(os.listdir(path=dir_path)) > 1 or len(os.listdir(path=dir_path)) == 0:
        raise Exception(
            f"len(os.listdir(path=dir_path)) > 1: {len(os.listdir(path=dir_path)) > 1}, \
             len(os.listdir(path=dir_path)) == 0: {len(os.listdir(path=dir_path)) == 0}"
        )
    else:
        tb_log_file_name = os.listdir(path=dir_path)[0]

    event_acc = EventAccumulator(f"{dir_path}/{tb_log_file_name}", size_guidance={'scalars': 0})
    event_acc.Reload() # ログファイルのサイズによっては非常に時間がかかる

    tb_data_dict = {}
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        tb_data_dict[tag] = [event.value for event in events]
    
    return tb_data_dict


def plot_and_save_img(
    model_names: List[str],
    tag: str,
    labels: List[str],
    ylabel: str,
    y_min: float,
    y_max: float,
    save_dir_path: str,
    img_name: str,
):
    """
    データをplotして画像の保存まで行う
    """
    tb_data_list = []
    for model_name in model_names:
        tb_data_list.append(get_tb_data_dict(model_name)[tag])
    
    plt.rcParams["font.size"] = 16        # fontのsize
    plt.rcParams["legend.framealpha"] = 1 # legendの透明度
    
    fig = plt.figure(figsize=(10, 6), dpi=144)
    for i, tb_data in enumerate(tb_data_list):
        plt.plot([i for i in range(len(tb_data))], tb_data, label=labels[i])
        print(f"i:{i}, length: {len(tb_data)}")

    plt.xlabel("更新回数")
    plt.ylabel(ylabel)
    plt.ylim(y_min, y_max)
    plt.grid()
    plt.legend()
    # plt.tight_layout()
    fig.savefig(f"{save_dir_path}/{img_name}", pad_inches=0.01)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_path",
        help="path to config file",
        type=str,
        default="./configs/audionav/av_nav/replica/plot_tb_data.yaml"
    )
    args = parser.parse_args()
    with open(args.yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # make directory to save
    now = datetime.datetime.now()
    save_dir_path = f"./imgs/plot_tb_data/{now.strftime('%Y_%m%d_%H%M%S')}"
    os.makedirs(save_dir_path, exist_ok=True)

    # save config
    with open(f"{save_dir_path}/config.yml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


    model_names = config["model_names"]
    tag = config["tag"]
    labels = config["labels"]
    ylabel = config["ylabel"]
    y_min = config["y_min"]
    y_max = config["y_max"]
    img_name = config["img_name"]

    plot_and_save_img(
        model_names=model_names,
        tag=tag,
        labels=labels,
        ylabel=ylabel,
        y_min=y_min,
        y_max=y_max,
        save_dir_path=save_dir_path,
        img_name=img_name,
    )

    print("FIN PLOT TENSORBOARD DATA !!")
    
