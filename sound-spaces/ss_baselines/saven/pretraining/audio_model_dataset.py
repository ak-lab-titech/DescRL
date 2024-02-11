
import os
import logging
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import fftconvolve
from skimage.measure import block_reduce
import pandas as pd

from ss_baselines.common.utils import to_tensor


class AudioDataset(Dataset):
    def __init__(self, scenes_sr_pairs, ooi_objects_id_name, ooi_regions_id_name, use_cache=False, split="train"):
        self.ooi_objects_id_name = ooi_objects_id_name
        self.ooi_regions_id_name = ooi_regions_id_name
        self.use_cache = use_cache

        self.files = list()
        self.goals = list()
        self.binaural_rir_dir = 'data/binaural_rirs/default'
        self.source_sound_dir = f'data/sounds/semantic_splits/{split}'
        self.source_sound_dict = dict()
        self.rir_sampling_rate = 16000
        sound_files = os.listdir(self.source_sound_dir)
        print("sound_files: ", sound_files)

        graph_filename = r"data/metadata/mp3d_graph_object.csv"
        df = pd.read_csv(graph_filename, delimiter=',')
        self.object_regions = {}
        for index, row in df.iterrows():
            obj = row['Sounding Objects']
            regions = row['Regions']
            regions = [reg.strip() for reg in regions.split(',')]
            self.object_regions[obj] = regions

        np.random.seed(42)
        for scene in tqdm(scenes_sr_pairs):
            goals = []
            for s, r in scenes_sr_pairs[scene]:
                sound_file = np.random.choice(sound_files)

                object_name, fileext = os.path.splitext(sound_file)
                object_id = list(self.ooi_objects_id_name.keys())[
                    list(self.ooi_objects_id_name.values()).index(object_name)]

                angle = np.random.choice([0, 90, 180, 270])
                rir_file = os.path.join(self.binaural_rir_dir, scene, str(angle), f"{r}_{s}.wav")

                self.files.append((rir_file, sound_file))
                goals.append(object_id)

            self.goals += goals

        self.data = [None] * len(self.goals)
        self.load_source_sounds()

    def audio_length(self, sound):
        return self.source_sound_dict[sound].shape[0] // self.rir_sampling_rate

    def load_source_sounds(self):
        sound_files = os.listdir(self.source_sound_dir)
        for sound_file in sound_files:
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, sound_file), sr=self.rir_sampling_rate)
            self.source_sound_dict[sound_file] = audio_data

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if (self.use_cache and self.data[item] is None) or not self.use_cache:
            rir_file, sound_file = self.files[item]
            audiogoal = self.compute_audiogoal(rir_file, sound_file)
            spectrogram = to_tensor(self.compute_spectrogram(audiogoal))

            object_id = self.goals[item]
            object_name = self.ooi_objects_id_name[object_id]
            regions = self.object_regions[object_name]

            regions_id = [list(self.ooi_regions_id_name.keys())[list(self.ooi_regions_id_name.values()).index(reg)]
                          for reg in regions]
            regions_id = torch.tensor([1 if reg_id in regions_id else 0
                                       for reg_id in range(len(self.ooi_regions_id_name))])

            # permute tensor to dimension [CHANNEL x HEIGHT X WIDTH]
            spectrogram = spectrogram.permute(2, 0, 1)

            inputs_outputs = (spectrogram, object_id, regions_id)

            if self.use_cache:
                self.data[item] = inputs_outputs
        else:
            inputs_outputs = self.data[item]

        return inputs_outputs

    def compute_audiogoal(self, binaural_rir_file, sound_file):
        sampling_rate = self.rir_sampling_rate
        try:
            sampling_freq, binaural_rir = wavfile.read(binaural_rir_file)  # float32
            # temp_file = 'binaural_rir_' + sound_file
            # wavfile.write(temp_file, sampling_rate, binaural_rir)
        except ValueError:
            logging.warning("{} file is not readable".format(binaural_rir_file))
            binaural_rir = np.zeros((sampling_rate, 2)).astype(np.float32)
        if len(binaural_rir) == 0:
            logging.debug("Empty RIR file at {}".format(binaural_rir_file))
            binaural_rir = np.zeros((sampling_rate, 2)).astype(np.float32)

        current_source_sound = self.source_sound_dict[sound_file]
        index = random.randint(0, self.audio_length(sound_file) - 2)
        if index * sampling_rate - binaural_rir.shape[0] < 0:
            source_sound = current_source_sound[: (index + 1) * sampling_rate]
            binaural_convolved = np.array([fftconvolve(source_sound, binaural_rir[:, channel]
                                                       ) for channel in range(binaural_rir.shape[-1])])
            audiogoal = binaural_convolved[:, index * sampling_rate: (index + 1) * sampling_rate]
        else:
            # include reverb from previous time step
            # Length of source_sound would be 1 sec + length of binaural_rir
            source_sound = current_source_sound[index * sampling_rate - binaural_rir.shape[0]
                                                : (index + 1) * sampling_rate]
            # source_sound = current_source_sound  # Convolve complete clip
            # Length of binaural_convolved would be 1 sec + 1 data point
            binaural_convolved = np.array([fftconvolve(source_sound, binaural_rir[:, channel], mode='valid',
                                                       ) for channel in range(binaural_rir.shape[-1])])
            # Length of audiogoal would be 1 sec
            audiogoal = binaural_convolved[:, :-1]

        # temp_file = 'audiogoal_' + sound_file
        # wavfile.write(temp_file, sampling_rate, audiogoal.T)

        return audiogoal

    @staticmethod
    def compute_spectrogram(audiogoal):
        def compute_stft(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
            stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft

        channel1_magnitude = np.log1p(compute_stft(audiogoal[0]))
        channel2_magnitude = np.log1p(compute_stft(audiogoal[1]))
        spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)

        return spectrogram
