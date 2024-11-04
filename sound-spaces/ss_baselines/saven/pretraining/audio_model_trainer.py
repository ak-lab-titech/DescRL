
import os
import sys
import pickle
import time
import logging
import copy
from collections import OrderedDict
from datetime import datetime
from itertools import product

sys.path.insert(0, "/home/4/ud02274/navigation/myss/sound-spaces")
sys.path.append("/home/4/ud02274/navigation/myss/habitat-lab")

import networkx as nx
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from soundspaces.utils import load_metadata
from utils import get_gpu_memory_map
from audio_model_dataset import AudioDataset
from audio_model_predictor import AudioPredictor


def get_scenes_sr_pairs(split):

    binaural_rir_dir = 'data/binaural_rirs/default'
    meta_dir = "data/metadata"

    scenes = os.listdir(binaural_rir_dir)

    scene_sr_pairs_split = {}
    for scene in tqdm(scenes):
        points, scene_graph = load_metadata(os.path.join(meta_dir, 'default', scene))

        subgraphs = list(nx.connected_components(scene_graph))

        sr_pairs = list()
        for subgraph in subgraphs:
            sr_pairs += list(product(subgraph, subgraph))
        np.random.seed(42)
        np.random.shuffle(sr_pairs)
        sr_pairs = sr_pairs[:50000]
        # sr_pairs = sr_pairs[:10]

        size = int(len(sr_pairs) * 0.80)
        if 'all' in split:
            scene_sr_pairs_split.setdefault(scene, sr_pairs)
        elif split == 'train':
            scene_sr_pairs_split.setdefault(scene, sr_pairs[:size])
        elif split == 'test':
            scene_sr_pairs_split.setdefault(scene, sr_pairs[size:])

    return scene_sr_pairs_split


class AudioPredictorTrainer:
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

        self.batch_size = 1024
        self.num_worker = 8
        self.lr = 1e-3
        self.weight_decay = None
        self.num_epoch = 50
        self.audio_predictor = AudioPredictor(self.num_objects, self.num_regions)
        summary(self.audio_predictor.predictor, input_size=(2, 65, 26), device='cpu')
        self.audio_predictor = nn.DataParallel(self.audio_predictor, device_ids=device_ids)
        self.audio_predictor.to(device=self.device)

    def run(self, splits, writer):

        datasets = dict()
        dataloaders = dict()
        dataset_sizes = dict()
        for split in splits:
            scenes_sr_pairs = get_scenes_sr_pairs(split)

            datasets[split] = AudioDataset(scenes_sr_pairs=scenes_sr_pairs, ooi_objects_id_name=self.ooi_objects_id_name,
                                           ooi_regions_id_name=self.ooi_regions_id_name, use_cache=True)
            dataloaders[split] = DataLoader(dataset=datasets[split], batch_size=self.batch_size, shuffle=True,
                                            pin_memory=True, num_workers=self.num_worker, sampler=None)

            dataset_sizes[split] = len(datasets[split])
            print('{} has {} samples'.format(split.upper(), dataset_sizes[split]))

        classifier_criterion_object = nn.CrossEntropyLoss().to(device=self.device)
        classifier_criterion_region = nn.BCELoss().to(device=self.device)
        model = self.audio_predictor
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)

        # training params
        since = time.time()
        best_acc_emr = 0
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
                    self.audio_predictor.train()  # Set model to training mode
                else:
                    self.audio_predictor.eval()  # Set model to evaluate mode

                running_total_loss = 0.0
                running_classifier_loss_objects = 0
                running_classifier_loss_regions = 0
                running_classifier_corrects_objects = 0
                running_exact_match_count_regions = 0
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

                    classifier_loss_objects = classifier_criterion_object(predicts[:, :self.num_objects],
                                                                          objects_id_gts.long())
                    classifier_loss_regions = classifier_criterion_region(predicts[:, -self.num_regions:],
                                                                          regions_id_gts)
                    loss = classifier_loss_objects + classifier_loss_regions

                    # backward + optimize only if in training phase
                    if split == 'train':
                        optimizer.zero_grad()  # zero the parameter gradients
                        loss.backward()
                        optimizer.step()

                    running_total_loss += loss.item() * predicts.size(0)
                    running_classifier_loss_objects += classifier_loss_objects.item() * objects_id_gts.size(0)
                    running_classifier_loss_regions += classifier_loss_regions.item() * regions_id_gts.size(0)

                    predicts_objects = predicts[:, :self.num_objects]
                    predicts_regions = predicts[:, -self.num_regions:]

                    # Object
                    _, predicts_objects = torch.max(predicts_objects, 1)

                    # Accuracy
                    running_classifier_corrects_objects += torch.sum(predicts_objects == objects_id_gts)

                    # Region
                    predicts_regions = torch.where(predicts_regions > 0.5, 1., 0.)

                    # Exact Match Ratio (EMR): Calculate the ratio of data instances for which the prediction is
                    # identical to its class label, over all data instances.

                    running_exact_match_regions = torch.all(predicts_regions == regions_id_gts, dim=1)
                    running_exact_match_count_regions += torch.sum(running_exact_match_regions)

                    # Hamming Loss: Average number of False Positives and False Negative

                    for j in range(regions_id_gts.size(0)):
                        running_hamming_loss_regions += torch.logical_xor(predicts_regions[j],
                                                                          regions_id_gts[j]).float()

                epoch_total_loss = running_total_loss / dataset_sizes[split]
                epoch_classifier_loss_objects = running_classifier_loss_objects / dataset_sizes[split]
                epoch_classifier_loss_regions = running_classifier_loss_regions / dataset_sizes[split]
                epoch_classifier_corrects_objects = running_classifier_corrects_objects / dataset_sizes[split]
                epoch_exact_match_ratio_regions = running_exact_match_count_regions / dataset_sizes[split]
                epoch_hamming_loss_regions = running_hamming_loss_regions / dataset_sizes[split]

                # Writing to tensorboard
                writer.add_scalar(f'Loss/{split}_total', epoch_total_loss, epoch)
                writer.add_scalar(f'Loss/{split}_classifier_objects', epoch_classifier_loss_objects, epoch)
                writer.add_scalar(f'Loss/{split}_classifier_regions', epoch_classifier_loss_regions, epoch)
                writer.add_scalar(f'Accuracy/{split}_objects', epoch_classifier_corrects_objects, epoch)
                writer.add_scalar(f'Exact_Match_Ratio/{split}_regions', epoch_exact_match_ratio_regions, epoch)
                for reg_id in range(self.num_regions):
                    writer.add_scalar(f'Hamming_Loss/regions/{split}/' + self.ooi_regions_id_name[reg_id],
                                      epoch_hamming_loss_regions[reg_id], epoch)

                # deep copy the model
                target_acc_emr = epoch_classifier_corrects_objects + epoch_exact_match_ratio_regions
                if 'test' in split and target_acc_emr > best_acc_emr:
                    best_acc_emr = target_acc_emr
                    best_model_wts = copy.deepcopy(model.state_dict())
                    self.save_checkpoint(splits, f"ckpt.{epoch}.pth")

                time_elapsed = time.time() - since
                logging.info('Results of {} split has on {} samples'.format(split.upper(), dataset_sizes[split]))
                logging.info('Time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                logging.info('Current Accuracy + Exact Match Ratio: {:4f}'.format(target_acc_emr))
                logging.info('Best Accuracy + Exact Match Ratio: {:4f}'.format(best_acc_emr))

        self.save_checkpoint(splits, f"best_test.pth", checkpoint={"audio_predictor": best_model_wts})

        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logging.info('Best Accuracy + Exact Match Ratio: {:4f}'.format(best_acc_emr))

        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)

    def save_checkpoint(self, splits, ckpt_path, checkpoint=None):

        train_mode = False
        for split in splits:
            if 'train' in split:
                train_mode = True

        if train_mode:
            if checkpoint is None:
                checkpoint = {"audio_predictor": self.audio_predictor.state_dict()}
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
        default='data/models/saven/audio',
        help="Modify model-dir",
    )
    parser.add_argument("--use-multiple-GPU", action="store_true", help="use multiple GPU")
    args = parser.parse_args()

    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(args.model_dir, 'tb', time_stamp)
    summary_writer = SummaryWriter(log_dir=log_dir)

    logging.basicConfig(filename=log_dir + os.sep + 'audio-model-trainer_' + time_stamp + '_logs.txt',
                        level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.info('log_dir: {}'.format(log_dir))

    audio_predictor_trainer = AudioPredictorTrainer(args.model_dir, args.use_multiple_GPU)

    if args.run_type == 'train':
        audio_predictor_trainer.run(['train', 'test'], summary_writer)
    elif args.run_type == 'train-all':
        audio_predictor_trainer.run(['train-all', 'test-all'], summary_writer)
    else:
        # ckpt = torch.load(os.path.join(args.model_dir, 'best_test.pth'))
        ckpt = torch.load(os.path.join(args.model_dir, 'best_test.pth'), map_location=audio_predictor_trainer.device)

        state_dict = ckpt['audio_predictor']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v

        audio_predictor_trainer.audio_predictor.load_state_dict(new_state_dict)
        # vision_predictor_trainer.audio_predictor.load_state_dict(ckpt['audio_predictor'])

        audio_predictor_trainer.run(['test', 'test-all'], summary_writer)


"""
python ss_baselines/saven/pretraining/audio_model_trainer.py --run-type train --model-dir data/models/saven_gt/audio
nohup python ss_baselines/saven/pretraining/audio_model_trainer.py --run-type train --model-dir data/models/saven_gt/audio > audio_model_trainer.txt &

"""
