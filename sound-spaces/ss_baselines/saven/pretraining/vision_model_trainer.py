
import argparse
import copy
import logging
import os
import sys
import pickle
import time
from collections import OrderedDict
from datetime import datetime

sys.path.insert(0, "/home/4/ud02274/navigation/myss/sound-spaces")
sys.path.append("/home/4/ud02274/navigation/myss/habitat-lab")

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from utils import get_gpu_memory_map
from vision_model_dataset import VisionDataset
from vision_model_predictor import VisionPredictor


def get_scenes_nodes(split):
    mp3d_scene_valid_semantic_nodes_filepath = r"data/metadata/mp3d_scene_valid_semantic_nodes.bin"

    with open(mp3d_scene_valid_semantic_nodes_filepath, 'rb') as fo:
        scene_valid_semantic_nodes = pickle.load(fo)

    if 'all' in split:
        return scene_valid_semantic_nodes

    scene_valid_semantic_nodes_split = {}
    for scene in scene_valid_semantic_nodes:

        size = int(len(scene_valid_semantic_nodes[scene]) * 0.80)
        np.random.seed(42)
        np.random.shuffle(scene_valid_semantic_nodes[scene])
        if split == 'train':
            scene_valid_semantic_nodes_split.setdefault(scene, scene_valid_semantic_nodes[scene][:size])
            # scene_valid_semantic_nodes_split.setdefault(scene, scene_valid_semantic_nodes[scene][:size][0:10])
        elif split == 'test':
            scene_valid_semantic_nodes_split.setdefault(scene, scene_valid_semantic_nodes[scene][size:])
            # scene_valid_semantic_nodes_split.setdefault(scene, scene_valid_semantic_nodes[scene][size:][0:5])

    return scene_valid_semantic_nodes_split


class VisionPredictorTrainer:
    def __init__(self, model_dir, use_multiple_gpu):
        self.model_dir = model_dir
        self.use_multiple_gpu = use_multiple_gpu

        mp3d_objects_of_interest_filepath = r"data/metadata/mp3d_objects_of_interest_data.bin"
        with open(mp3d_objects_of_interest_filepath, 'rb') as bin_file:
            self.ooi_objects_id_name = pickle.load(bin_file)
            self.ooi_regions_id_name = pickle.load(bin_file)

        self.num_objects = len(self.ooi_objects_id_name)
        self.num_regions = len(self.ooi_regions_id_name)

        # Finding and using least used GPUs
        gpus_memory = get_gpu_memory_map()
        device_ids = [k for k, v in sorted(gpus_memory.items(), key=lambda x: x[1], reverse=False)]
        if not use_multiple_gpu:
            device_ids = [device_ids[0]]

        self.device = (torch.device("cuda", device_ids[0]))
        logging.info('GPU IDs: {}'.format(device_ids))

        self.batch_size = 512  # 512, 256
        self.num_worker = 8
        self.lr = 1e-3
        self.weight_decay = None
        self.num_epoch = 50
        self.vision_predictor = VisionPredictor(self.num_objects, self.num_regions)
        summary(self.vision_predictor.predictor, input_size=(3, 128, 128), device='cpu')
        self.vision_predictor = nn.DataParallel(self.vision_predictor, device_ids=device_ids)
        self.vision_predictor.to(device=self.device)

    def run(self, splits, writer):

        datasets = dict()
        dataloaders = dict()
        dataset_sizes = dict()
        for split in splits:
            scenes_nodes = get_scenes_nodes(split)

            datasets[split] = VisionDataset(scenes_nodes, self.num_objects, self.num_regions)
            dataloaders[split] = DataLoader(dataset=datasets[split], batch_size=self.batch_size, shuffle=True,
                                            pin_memory=True, num_workers=self.num_worker, sampler=None)

            dataset_sizes[split] = len(datasets[split])
            print('{} has {} samples'.format(split.upper(), dataset_sizes[split]))

        classifier_criterion = nn.BCELoss().to(device=self.device)
        model = self.vision_predictor
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)

        # training params
        since = time.time()
        best_emr = 0
        best_model_wts = None

        num_epoch = 1
        for split in splits:
            if 'train' in split:
                num_epoch = self.num_epoch

        for epoch in range(num_epoch):
            logging.info('-' * 40)
            logging.info('Epoch {}/{}'.format(epoch + 1, num_epoch))

            # Each epoch has a training and validation phase
            for split in splits:
                if 'train' in split:
                    self.vision_predictor.train()  # Set model to training mode
                else:
                    self.vision_predictor.eval()  # Set model to evaluate mode

                running_total_loss = 0
                running_classifier_loss_objects = 0
                running_classifier_loss_regions = 0
                running_exact_match_count_objects = 0
                running_exact_match_count_regions = 0
                running_hamming_loss_objects = torch.zeros(self.num_objects).to(device=self.device)
                running_hamming_loss_regions = torch.zeros(self.num_regions).to(device=self.device)

                # Iterating over data once is one epoch
                for i, data in enumerate(tqdm(dataloaders[split])):
                    # get the inputs
                    inputs, objects_id_gts, regions_id_gts = data

                    inputs = inputs.to(device=self.device, dtype=torch.float)
                    objects_id_gts = objects_id_gts.to(device=self.device, dtype=torch.float)
                    regions_id_gts = regions_id_gts.to(device=self.device, dtype=torch.float)

                    # forward
                    predicts = model(inputs)

                    classifier_loss_objects = classifier_criterion(predicts[:, :self.num_objects], objects_id_gts)
                    classifier_loss_regions = classifier_criterion(predicts[:, -self.num_regions:], regions_id_gts)
                    loss = classifier_loss_objects + classifier_loss_regions

                    # backward + optimize only if in training phase
                    if 'train' in split:
                        optimizer.zero_grad()  # zero the parameter gradients
                        loss.backward()
                        optimizer.step()

                    running_total_loss += loss.item() * predicts.size(0)
                    running_classifier_loss_objects += classifier_loss_objects.item() * objects_id_gts.size(0)
                    running_classifier_loss_regions += classifier_loss_regions.item() * regions_id_gts.size(0)

                    predicts = torch.where(predicts > 0.5, 1., 0.)
                    predicts_objects = predicts[:, :self.num_objects]
                    predicts_regions = predicts[:, -self.num_regions:]

                    # Exact Match Ratio (EMR): Calculate the ratio of data instances for which the prediction is
                    # identical to its class label, over all data instances.

                    running_exact_match_objects = torch.all(predicts_objects == objects_id_gts, dim=1)
                    running_exact_match_count_objects += torch.sum(running_exact_match_objects)
                    running_exact_match_regions = torch.all(predicts_regions == regions_id_gts, dim=1)
                    running_exact_match_count_regions += torch.sum(running_exact_match_regions)

                    # Hamming Loss: Average number of False Positives and False Negative

                    for j in range(objects_id_gts.size(0)):
                        running_hamming_loss_objects += torch.logical_xor(predicts_objects[j],
                                                                          objects_id_gts[j]).float()
                    for j in range(regions_id_gts.size(0)):
                        running_hamming_loss_regions += torch.logical_xor(predicts_regions[j],
                                                                          regions_id_gts[j]).float()

                epoch_total_loss = running_total_loss / dataset_sizes[split]
                epoch_classifier_loss_objects = running_classifier_loss_objects / dataset_sizes[split]
                epoch_classifier_loss_regions = running_classifier_loss_regions / dataset_sizes[split]
                epoch_exact_match_ratio_objects = running_exact_match_count_objects / dataset_sizes[split]
                epoch_exact_match_ratio_regions = running_exact_match_count_regions / dataset_sizes[split]
                epoch_hamming_loss_objects = running_hamming_loss_objects / dataset_sizes[split]
                epoch_hamming_loss_regions = running_hamming_loss_regions / dataset_sizes[split]

                # Writing to tensorboard
                writer.add_scalar(f'Loss/{split}_total', epoch_total_loss, epoch)
                writer.add_scalar(f'Loss/{split}_classifier_objects', epoch_classifier_loss_objects, epoch)
                writer.add_scalar(f'Loss/{split}_classifier_regions', epoch_classifier_loss_regions, epoch)
                writer.add_scalar(f'Exact_Match_Ratio/{split}_objects', epoch_exact_match_ratio_objects, epoch)
                writer.add_scalar(f'Exact_Match_Ratio/{split}_regions', epoch_exact_match_ratio_regions, epoch)
                for obj_id in range(self.num_objects):
                    writer.add_scalar(f'Hamming_Loss/objects/{split}/' + self.ooi_objects_id_name[obj_id],
                                      epoch_hamming_loss_objects[obj_id], epoch)
                for reg_id in range(self.num_regions):
                    writer.add_scalar(f'Hamming_Loss/regions/{split}/' + self.ooi_regions_id_name[reg_id],
                                      epoch_hamming_loss_regions[reg_id], epoch)

                # deep copy the model
                target_emr = epoch_exact_match_ratio_objects + epoch_exact_match_ratio_regions
                if 'test' in split and target_emr > best_emr:
                    best_emr = target_emr
                    best_model_wts = copy.deepcopy(model.state_dict())
                    self.save_checkpoint(splits, f"ckpt.{epoch}.pth")

                time_elapsed = time.time() - since
                logging.info('Results of {} split has on {} samples'.format(split.upper(), dataset_sizes[split]))
                logging.info('Time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                logging.info('Current Exact Match Ratio: {:4f}'.format(target_emr))
                logging.info('Best Exact Match Ratio: {:4f}'.format(best_emr))

        self.save_checkpoint(splits, f"best_test.pth", checkpoint={"vision_predictor": best_model_wts})

        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logging.info('Best Exact Match Ratio: {:4f}'.format(best_emr))

        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)

    def save_checkpoint(self, splits, ckpt_path, checkpoint=None):

        train_mode = False
        for split in splits:
            if 'train' in split:
                train_mode = True

        if train_mode:
            if checkpoint is None:
                checkpoint = {"vision_predictor": self.vision_predictor.state_dict()}
            torch.save(checkpoint, os.path.join(self.model_dir, ckpt_path))


if __name__ == '__main__':

    print("Current working directory: {0}".format(os.getcwd()))
    os.chdir('../sound-spaces')
    print("Current working directory: {0}".format(os.getcwd()))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "test", "train-all"],
        required=True,
        help="run type of the experiment (train, test, or train-all)",
    )
    parser.add_argument(
        "--model-dir",
        default='data/models/saven/vision',
        help="Modify model-dir",
    )
    parser.add_argument("--use-multiple-GPU", action="store_true", help="use multiple GPU")
    args = parser.parse_args()

    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(args.model_dir, 'tb', time_stamp)
    summary_writer = SummaryWriter(log_dir=log_dir)

    logging.basicConfig(filename=log_dir + os.sep + 'vision-model-trainer_' + time_stamp + '_logs.txt',
                        level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.info('log_dir: {}'.format(log_dir))

    vision_predictor_trainer = VisionPredictorTrainer(args.model_dir, args.use_multiple_GPU)

    if args.run_type == 'train':
        vision_predictor_trainer.run(['train', 'test'], summary_writer)
    elif args.run_type == 'train-all':
        vision_predictor_trainer.run(['train-all', 'test-all'], summary_writer)
    else:
        # ckpt = torch.load(os.path.join(args.model_dir, 'best_test.pth'))
        ckpt = torch.load(os.path.join(args.model_dir, 'best_test.pth'), map_location=vision_predictor_trainer.device)

        state_dict = ckpt['vision_predictor']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v

        vision_predictor_trainer.vision_predictor.load_state_dict(new_state_dict)
        # vision_predictor_trainer.vision_predictor.load_state_dict(ckpt['vision_predictor'])

        vision_predictor_trainer.run(['test', 'test-all'], summary_writer)

"""
python ss_baselines/saven/pretraining/vision_model_trainer.py --run-type train --model-dir data/models/saven_gt/vision
nohup python ss_baselines/saven/pretraining/vision_model_trainer.py --run-type train --model-dir data/models/saven_gt/vision > vision_model_trainer.txt &

"""
