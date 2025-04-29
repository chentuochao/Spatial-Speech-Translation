"""
Torch dataset object for synthetically rendered
spatial data
"""
import json
import random

from typing import Tuple
from pathlib import Path

import torch
import numpy as np
import os

import src.utils as utils

import pandas as pd
import torchaudio
import torchaudio.transforms as AT
import librosa
from scipy.signal import resample, convolve
from src.datasets.augments import generate_white_noise, generate_pink_noise, generate_brown_noise

SOUND_SPEED = 341.0


def conver_Az2TDoA(az, head_size, sr):
    az = np.deg2rad(az)
    return (head_size*np.sin(az))/SOUND_SPEED * sr

def conver_TDoA2Az(sample, head_size, sr):
    sin_az = sample/sr * SOUND_SPEED / head_size
    if sin_az > 1:
        sin_az = 1
    if sin_az < -1:
        sin_az = -1

    az = np.rad2deg(np.arcsin(sin_az))
    return az

def roll_array(wav, TDoA):
    if TDoA == 0:
        return wav
    elif TDoA > 0:
        wav[1, :] = torch.roll(wav[1, :], shifts = -TDoA, dims = 0)
        wav[1, -TDoA:] = 0
    else:
        wav[1, :] = torch.roll(wav[1, :], shifts = -TDoA, dims = 0)
        wav[1, :-TDoA] = 0
    return wav

class Dataset(torch.utils.data.Dataset):
    """
    Dataset of mixed waveforms and their corresponding ground truth waveforms
    recorded at different microphone.

    Data format is a pair of Tensors containing mixed waveforms and
    ground truth waveforms respectively. The tensor's dimension is formatted
    as (n_microphone, duration).

    Each scenario is represented by a folder. Multiple datapoints are generated per
    scenario. This can be customized using the points_per_scenario parameter.
    """
    def __init__(self, input_dir, n_mics=2, sr=16000,
                 prob_neg = 0.4, sig_len = 5, head_size = 0.18, BG_prob = 0.4,
                 split = 'train', aug=False,
                 max_white_noise_level=1e-2, max_pink_noise_level=3e-2, max_brown_noise_level = 2e-2
                 ):
        super().__init__()

        self.aug = aug
        self.max_white_noise_level = max_white_noise_level
        self.max_pink_noise_level = max_pink_noise_level
        self.max_brown_noise_level = max_brown_noise_level

        self.dirs = sorted(list(Path(input_dir).glob('[0-9]*')))
        TDoA_bound = int(head_size/SOUND_SPEED*sr)
        self.head_size = head_size
        self.TDoA_options = [i for i in range(-TDoA_bound, TDoA_bound + 1)]
        # Physical params
        self.n_mics = n_mics
        self.sig_len = int(sig_len*sr)
        self.sr = sr
        # Data augmentation
        self.split = split
        self.prob_neg = prob_neg
        self.BG_prob = BG_prob

        self.valid_dis = []
        for dir0 in self.dirs:
            metadata = utils.read_json(os.path.join(dir0, 'metadata.json'))
            spk_info = metadata["infos"]
            if np.abs(spk_info[0]["pos"][0] - spk_info[1]["pos"][0]) <= 10:
                continue
            self.valid_dis.append(dir0)
        self.dirs = self.valid_dis

        self.val_len = int(len(self.dirs) // 8) * 8

    def __len__(self) -> int:
        return self.val_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mixed_data - M x T
            target_voice_data - M x T
            window_idx_one_hot - 1-D
        """
        #print("Fetch idx: ", idx)
        
        curr_dir = self.dirs[idx%len(self.dirs)]
        return self.get_mixture_and_gt(curr_dir)



    def sample_positive(self, mixture, spk_wav, spk_TDoA):
        spk_TDoA_int = np.round(spk_TDoA).astype(int)
        idxs = [i for i in range(0, len(spk_TDoA))]
        target_id = np.random.choice(idxs)
        ## find the source which is most close to it
        TDOA = spk_TDoA_int[target_id]
        target_id = np.argmin(np.abs((np.array(spk_TDoA) - TDOA)))
        # print(target_id)
        gt = spk_wav[target_id]        
        mixture = roll_array(mixture, TDOA)

        return mixture, gt, TDOA

    def sample_negative(self,mixture, spk_wav, spk_TDoA):
        spk_TDoA_int = np.round(spk_TDoA).astype(int)
        neg_options = []
        for s in self.TDoA_options:
            if s not in spk_TDoA_int:
                neg_options.append(s)
        TDOA_neg = np.random.choice(neg_options)

        gt = torch.zeros_like(mixture)

        mixture = roll_array(mixture, TDOA_neg)

        return mixture, gt, TDOA_neg

    def get_mixture_and_gt(self, curr_dir):
        # Get metadata
        metadata = utils.read_json(os.path.join(curr_dir, 'metadata.json'))

        spk_info = metadata["infos"]

        # calculate the gt sample offset
        spk_TDoA = []
        spk_wav = []


        BG = utils.read_audio_file_torch(os.path.join(curr_dir, 'BG.wav'), 1)

        for spk in spk_info:
            # print(spk["pos"])
            pos = spk["pos"]
            fname = spk["name"]
            az = np.deg2rad(pos[0])
            TDoA = (self.head_size*np.sin(az))/SOUND_SPEED * self.sr
            wav = utils.read_audio_file_torch(os.path.join(curr_dir, fname + '.wav'), 1)
            spk_wav.append(wav)
            spk_TDoA.append(TDoA)
        
        if np.random.rand() > self.BG_prob:
            mixture = spk_wav[0] + spk_wav[1]
        else:
            mixture = spk_wav[0] + spk_wav[1] + BG

        if self.aug and self.split == 'train':
            white_noise = \
                generate_white_noise(mixture.shape, self.max_white_noise_level).float()
            pink_noise = \
                generate_pink_noise(mixture.shape, self.max_pink_noise_level).float()
            brown_noise = \
                generate_brown_noise(mixture.shape, self.max_brown_noise_level).float()
            # print("add noise!!!!!", torch.sum(mixture.abs()), torch.sum(white_noise.abs()), torch.sum(pink_noise.abs()), torch.sum(brown_noise.abs()))
            mixture = mixture + (white_noise + pink_noise + brown_noise)

        sample_pos = True
        if np.random.rand() < self.prob_neg:
            mixture, gt, TDOA = self.sample_negative(mixture, spk_wav, spk_TDoA)
            sample_pos = False
        else:
            mixture, gt, TDOA = self.sample_positive(mixture, spk_wav, spk_TDoA)
            sample_pos = True


        ### padding and crop 
        gt0 = spk_wav[0] 
        gt1 = spk_wav[1] 
        L = mixture.shape[-1]
        if  L > self.sig_len:
            start_idx = np.random.randint(low=0, high=L - self.sig_len)
            mixture = mixture[:, start_idx : start_idx + self.sig_len]
            gt = gt[:, start_idx : start_idx + self.sig_len]
            gt0 = gt0[:, start_idx : start_idx + self.sig_len]
            gt1 = gt1[:, start_idx : start_idx + self.sig_len]

        elif L <  self.sig_len:
            pad_width = self.sig_len - L
            mixture = torch.nn.functional.pad(mixture, (0, pad_width))
            gt = torch.nn.functional.pad(gt, (0, pad_width))
            gt0 = torch.nn.functional.pad(gt0, (0, pad_width))
            gt1 = torch.nn.functional.pad(gt1, (0, pad_width))
    
        if torch.max(torch.abs(gt))  < 9e-5:# or torch.mean(torch.abs(gt))  < 5e-5:
            sample_pos = False
        else:
            sample_pos = True


        inputs = {
            'mixture': mixture.float(),
        }
        
        targets = {
            'target':gt.float(),
            'sample_pos': sample_pos,
            'TDoA_gt': spk_TDoA,
            'other':[gt0.float(), gt1.float()],
        }

        return inputs, targets

   


class InferDataset(torch.utils.data.Dataset):
    """
    Dataset of mixed waveforms and their corresponding ground truth waveforms
    recorded at different microphone.

    Data format is a pair of Tensors containing mixed waveforms and
    ground truth waveforms respectively. The tensor's dimension is formatted
    as (n_microphone, duration).

    Each scenario is represented by a folder. Multiple datapoints are generated per
    scenario. This can be customized using the points_per_scenario parameter.
    """
    def __init__(self, input_dir, n_mics=2, sr=16000,
                 prob_neg = 0.4, sig_len = 5, head_size = 0.18, BG_prob = 0.4,
                 split = 'train'):
        super().__init__()
        self.dirs = sorted(list(Path(input_dir).glob('[0-9]*')))
        TDoA_bound = int(head_size/SOUND_SPEED*sr)
        self.head_size = head_size
        self.TDoA_options = [i for i in range(-TDoA_bound, TDoA_bound + 1)]
        # Physical params
        self.n_mics = n_mics
        self.sig_len = int(sig_len*sr)
        self.sr = sr
        # Data augmentation
        self.split = split
        self.prob_neg = prob_neg
        self.BG_prob = BG_prob

        self.valid_dis = []
        for dir0 in self.dirs:
            metadata = utils.read_json(os.path.join(dir0, 'metadata.json'))
            spk_info = metadata["infos"]
            if np.abs(spk_info[0]["pos"][0] - spk_info[1]["pos"][0]) <= 10:
                continue
            self.valid_dis.append(dir0)
        self.dirs = self.valid_dis

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mixed_data - M x T
            target_voice_data - M x T
            window_idx_one_hot - 1-D
        """
        #print("Fetch idx: ", idx)
        
        curr_dir = self.dirs[idx%len(self.dirs)]
        return self.get_mixture_and_gt(curr_dir)


    def get_mixture_and_gt(self, curr_dir):
        # Get metadata
        metadata = utils.read_json(os.path.join(curr_dir, 'metadata.json'))

        spk_info = metadata["infos"]

        # calculate the gt sample offset
        spk_angle = []
        spk_TDoA = []
        spk_wav = []
        spk_name = []
        spk_lang = []

        BG = utils.read_audio_file_torch(os.path.join(curr_dir, 'BG.wav'), 1)

        for spk in spk_info:
            pos = spk["pos"]
            fname = spk["name"]
            spk_name.append(fname)
            spk_angle.append(pos[0])
            az = np.deg2rad(pos[0])
            spk_lang.append(spk["lang"])
            TDoA = (self.head_size*np.sin(az))/SOUND_SPEED * self.sr
            wav = utils.read_audio_file_torch(os.path.join(curr_dir, fname + '.wav'), 1)
            spk_wav.append(wav)
            spk_TDoA.append(TDoA)

        ### padding and crop
        gt0 = spk_wav[0]
        gt1 = spk_wav[1]    
        use_BG = False
        if np.random.rand() > self.BG_prob:
            mixture = spk_wav[0] + spk_wav[1]
            use_BG = False
        else:
            mixture = spk_wav[0] + spk_wav[1] + BG
            use_BG = True
 
        # L = mixture.shape[-1]
        # if  L > self.sig_len:
        #     start_idx = np.random.randint(low=0, high=L - self.sig_len)
        #     mixture = mixture[:, start_idx : start_idx + self.sig_len]
        #     gt0 = gt0[:, start_idx : start_idx + self.sig_len]
        #     gt1 = gt1[:, start_idx : start_idx + self.sig_len]

        # elif L <  self.sig_len:
        #     pad_width = self.sig_len - L
        #     mixture = torch.nn.functional.pad(mixture, (0, pad_width))
        #     gt0 = torch.nn.functional.pad(gt0, (0, pad_width))
        #     gt1 = torch.nn.functional.pad(gt1, (0, pad_width))


        inputs = {
            'mixture': mixture.float(),
        }
        
        targets = {
            'target':[gt0.float(), gt1.float()],
            'angle_gt': spk_angle,
            'TDoA_gt': spk_TDoA,
            "names": spk_name,
            "langs": spk_lang,
            'BG': use_BG
        }

        return inputs, targets







class InferDataset_Speechmatrix(torch.utils.data.Dataset):
    """
    Dataset of mixed waveforms and their corresponding ground truth waveforms
    recorded at different microphone.

    Data format is a pair of Tensors containing mixed waveforms and
    ground truth waveforms respectively. The tensor's dimension is formatted
    as (n_microphone, duration).

    Each scenario is represented by a folder. Multiple datapoints are generated per
    scenario. This can be customized using the points_per_scenario parameter.
    """
    def __init__(self, input_dir, lang_folder ,n_mics=2, sr=16000,
                 prob_neg = 0.4, sig_len = 5, head_size = 0.18, BG_prob = 0.4,
                 split = 'train'):
        super().__init__()

        self.all_lang_files = []
        voice_path = lang_folder + "clips/"
        voice_tsv = lang_folder + f"{split}.tsv"
        file_list = pd.read_csv(voice_tsv, sep='\t')['client_id'].tolist()

        for fname in file_list:
            if "speechmatrix" in fname:
                self.all_lang_files.append(voice_path + fname)
        print(voice_tsv, len(self.all_lang_files))

        self.dirs = sorted(list(Path(input_dir).glob('[0-9]*')))
        TDoA_bound = int(head_size/SOUND_SPEED*sr)
        self.head_size = head_size
        self.TDoA_options = [i for i in range(-TDoA_bound, TDoA_bound + 1)]
        # Physical params
        self.n_mics = n_mics
        self.sig_len = int(sig_len*sr)
        self.sr = sr
        # Data augmentation
        self.split = split
        self.prob_neg = prob_neg
        self.BG_prob = BG_prob

        self.valid_dis = []
        for dir0 in self.dirs:
            metadata = utils.read_json(os.path.join(dir0, 'metadata.json'))
            spk_info = metadata["infos"]
            if np.abs(spk_info[0]["pos"][0] - spk_info[1]["pos"][0]) <= 10:
                continue
            self.valid_dis.append(dir0)
        self.dirs = self.valid_dis

    def __len__(self) -> int:
        return min([len(self.dirs), len(self.all_lang_files)//2])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mixed_data - M x T
            target_voice_data - M x T
            window_idx_one_hot - 1-D
        """
        #print("Fetch idx: ", idx)
        
        curr_dir = self.dirs[idx%len(self.dirs)]
        return self.get_mixture_and_gt(idx, curr_dir)

    def pad_to_max(self, audio1, audio2):
        L1 = audio1.shape[-1]
        L2 = audio2.shape[-1]

        audio1 = audio1/np.abs(audio1).max()
        audio2 = audio2/np.abs(audio2).max()

        if L1 > L2:
            audio2 = np.pad(audio2, (0, L1 - L2))
        elif L1 < L2:
            audio1 = np.pad(audio1, (0, L2 - L1))

        return audio1, audio2

    def get_mixture_and_gt(self, idx, curr_dir):
        # Get metadata
        metadata = utils.read_json(os.path.join(curr_dir, 'metadata.json'))

        spk_info = metadata["infos"]

        # calculate the gt sample offset
        spk_angle = []
        spk_TDoA = []
        spk_wav = []
        spk_name = []
        spk_lang = []

        BG = utils.read_audio_file_torch(os.path.join(curr_dir, 'BG.wav'), 1)
        rng = np.random.RandomState(idx)

        ## get wav file
        sample_names = []
        sample_dirs = []
        sample_langs = []
        sample_audio = []
        while True:
            _id =  random.randint(0, len(self.all_lang_files) - 1)
            sample_dir = self.all_lang_files[_id]
    
            sample_name = sample_dir.split('/')[-1]

            if sample_name in sample_names:
                continue

            if not os.path.exists(sample_dir):
                print(f"Warning !!!! {sample_dir} not exist", )
                continue

            if os.path.getsize(sample_dir) == 0:
                print(f"empty file", sample_dir)
                continue

            audio, sr = librosa.load(sample_dir, sr=self.sr)
            if audio.shape[-1] < self.sr * 1:
                print("Warining!!!, too short audio discard", audio.shape[-1]/sr)
                continue
            sample_audio.append(audio)
            # audio2, sr2 = librosa.load(sample_dirs[1], sr=None)

            sample_dirs.append(sample_dir)
            sample_names.append(sample_name.split(".")[0])

            if len(sample_dirs) >= 2:
                break
        audio, audio2 = sample_audio[0], sample_audio[1]
        audio, audio2 = self.pad_to_max(audio, audio2)
        audio *= rng.uniform(low = 0.6, high = 0.9)
        audio2 *= rng.uniform(low = 0.6, high = 0.9)


        sample_audio = [audio, audio2]
        spk_wav = []
        spk_name = sample_names

        for _i, spk in enumerate(spk_info):
            pos = spk["pos"]
            fname = spk["name"]
            # spk_name.append(fname)
            spk_angle.append(pos[0])
            az = np.deg2rad(pos[0])
            spk_lang.append(spk["lang"])
            TDoA = (self.head_size*np.sin(az))/SOUND_SPEED * self.sr
            rir = np.load(os.path.join(curr_dir, fname + '.npy'))

            src = sample_audio[_i]
            src_l = convolve(src, rir[0])[:len(src)]
            src_r = convolve(src, rir[1])[:len(src)]
            bi_src = np.stack([src_l, src_r], axis=0)
            spk_wav.append(bi_src)
            spk_TDoA.append(TDoA)

        ### normalize
        gt0 = torch.from_numpy(spk_wav[0])
        gt1 = torch.from_numpy(spk_wav[1])
        use_BG = False
        if np.random.rand() > self.BG_prob:
            mixture = gt0 + gt1
            use_BG = False
        else:
            noise_scale = rng.uniform(low = 0.05, high = 0.5)
            BG = BG/torch.abs(BG).max() * torch.abs(gt1).max()
            BG = BG * noise_scale
            print(gt0.shape, gt1.shape, BG.shape)
            if BG.shape[-1] >= gt1.shape[-1]:
                BG = BG[:, :gt1.shape[-1]]
            else:
                BGs = torch.cat([BG for _ in range( int(gt1.shape[-1]//BG.shape[-1]) + 1 ) ], dim = -1)
                BG = BGs[:, :gt1.shape[-1]]
            mixture = gt0 + gt1+ BG
            use_BG = True
 
        norm_factor = torch.abs(mixture).max()/0.9
        gt0 /= norm_factor
        gt1 /= norm_factor
        mixture /= norm_factor

        inputs = {
            'mixture': mixture.float(),
        }
        
        targets = {
            'target':[gt0.float(), gt1.float()],
            'angle_gt': spk_angle,
            'TDoA_gt': spk_TDoA,
            "names": spk_name,
            "langs": spk_lang,
            'BG': use_BG
        }

        return inputs, targets
