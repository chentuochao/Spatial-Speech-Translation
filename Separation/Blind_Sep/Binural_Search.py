import numpy as np 
import torch
from asteroid.losses.sdr import SingleSrcNegSDR
import math
import src.utils as utils
import os 

SOUND_SPEED = 341.0


sisdrloss = SingleSrcNegSDR('sisdr')

from scipy.ndimage import uniform_filter1d
import librosa

def si_sdr(estimated_signal, reference_signals, scaling=True):
    """
    This is a scale invariant SDR. See https://arxiv.org/pdf/1811.02508.pdf
    or https://github.com/sigsep/bsseval/issues/3 for the motivation and
    explanation

    Input:
        estimated_signal and reference signals are (N,) numpy arrays

    Returns: SI-SDR as scalar
    """
    MIN_ERR = 1e-8
    Rss = np.dot(reference_signals, reference_signals)
    this_s = reference_signals

    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum() + MIN_ERR

    SDR = 10 * math.log10(Sss/Snn)

    return SDR

def check_sisnr_win(sisnr_list, SISNR_THRESHOLD = -2, SISNR_THRESHOLD2 = -8): # -2. -6
    
    SAME_FLAG = False
    SAME_FLAG2 = True
    for value in sisnr_list:
        if value > SISNR_THRESHOLD:
            SAME_FLAG = True
        if value < SISNR_THRESHOLD2:
            SAME_FLAG2 = False

    return (SAME_FLAG and SAME_FLAG2)



def split_wav(wav, top_db = 18):
    MIN_SEG = 2000 #800
    MAX_SEG = 6000
    power_list = librosa.feature.rms(y = wav, frame_length=1024, hop_length=256)
    max_ref = np.amax(power_list)
    split_threshold =  0.04
    if max_ref <split_threshold:
        #print("Max apmlitude of signal is too low and reset the ref max value!")
        intervals = librosa.effects.split(wav, top_db=top_db, ref = split_threshold, frame_length=1024, hop_length=256)
    else:
        intervals = librosa.effects.split(wav, top_db=top_db,  frame_length=1024, hop_length=256)

    finetune_seg = []
    for indexes in intervals:
        interval_len = indexes[1] - indexes[0]
        if interval_len < MIN_SEG:
            continue
        elif interval_len > MAX_SEG :
            num_seg = interval_len//MAX_SEG
            for i in range(num_seg):
                if i >= num_seg - 1:
                    finetune_seg.append([indexes[0] + i*MAX_SEG, indexes[1]])
                else:
                    finetune_seg.append([indexes[0] + i*MAX_SEG, indexes[0] + (i+1)*MAX_SEG ])
        else:
            finetune_seg.append([indexes[0], indexes[1]])

    return finetune_seg

def split_wise_sisdr(estimated_signal, reference_signals, seg_index):
    assert(len(seg_index) > 0)
    sisdr_list = []
    for range_index in seg_index:
        seg1 = estimated_signal[0, range_index[0]: range_index[1]]
        seg2 = reference_signals[0, range_index[0]: range_index[1]]
        similarity = si_sdr(seg1, seg2)
        sisdr_list.append(similarity)
    
    return sisdr_list


def max_avg_power(x: np.ndarray, window_size: int = 12000):
    max_avg_energy = uniform_filter1d(x**2, size=window_size, mode='constant', origin=-window_size//2)	
    max_avg_energy = np.sqrt(np.abs(max_avg_energy))
    return max_avg_energy.max()

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

class Binural_Search(object):
    def __init__(self, sr, model, device, head_size=0.18):
        TDoA_bound = int(head_size/SOUND_SPEED*sr)
        self.TDoA_options = [i for i in range(-TDoA_bound - 1, TDoA_bound + 2)]
        self.model = model
        self.model.eval()
        self.device = device
    


    def process_gt(self, wav, TDoAs):
        wave_all = []
        for TDoA in TDoAs:
            TDoA = int(round(TDoA))
            shifted_wav = roll_array(torch.clone(wav), TDoA)
            wave_all.append(shifted_wav)
        with torch.no_grad():
            input_wav = torch.stack(wave_all, dim = 0) 
            #print(input_wav.shape)
            inputs = {}
            inputs['mixture'] = input_wav.to(self.device)
            out = self.model(inputs)['output']

        #audios = []
        #for i in range(0, out.shape[0]):
        #    audios.append(out[i, :].cpu())

        return out[:, 0, :].cpu() #audios

    def process_search(self, wav):
        wave_all = []
        for TDoA in self.TDoA_options:
            shifted_wav = roll_array(torch.clone(wav), TDoA)
            wave_all.append(shifted_wav)
        BATCH_SIZE = len(wave_all)//2
        idx = 0
        out = []
        with torch.no_grad():
            while True:
                if idx >= len(wave_all):
                    break
                elif idx + BATCH_SIZE >= len(wave_all):
                    input_wav = torch.stack(wave_all[idx:], dim = 0) 
                    idx = len(wave_all)
                else:
                    input_wav = torch.stack(wave_all[idx:idx+BATCH_SIZE], dim = 0) 
                    idx += BATCH_SIZE
                inputs = {}
                inputs['mixture'] = input_wav.to(self.device)
                out0 = self.model(inputs)['output']
                out.append(out0)
                
        out = torch.cat(out, dim = 0) 
        assert(len(self.TDoA_options) == out.shape[0])
        powers = []
        power2s = []
        sigs_all = []
        sigs_all_binural = []

        for i in range(out.shape[0]):
            sig = out[i].cpu().numpy()
            sig0 = sig[0:1]
            power2 = max_avg_power(sig0) 
            power = np.sqrt(np.mean(sig0**2)) 

            ## append to lists
            powers.append(power)
            power2s.append(power2)
            sigs_all.append(sig0)
            sigs_all_binural.append(sig)

        sort_idx = np.argsort(-1*np.array(powers))
        SI_SDR_THRESHOLD = -4
        
        clusters = {}
        audio_unique = []
        for _id in sort_idx:
            unique = True
            if power2s[_id] < 1.5e-2 or powers[_id] < 8e-3: #power2s[_id] < 2e-2 or powers[_id] < 1e-2:
                continue 
            sim_lists = []
            audio1 = sigs_all[_id] # only take the first channel
            seg_win = split_wav(audio1)
            if(len(seg_win) == 0):
                print("discard because no invalid split!!!")
                continue
            # print(_id, self.TDoA_options[_id])
            for cluster_id in clusters:
                final_candidate_id = clusters[cluster_id][0]
                audio2 = sigs_all[final_candidate_id]

                sisdr_list = split_wise_sisdr(audio1, audio2, seg_win)
                # print(sisdr_list)
                check_valid = check_sisnr_win(sisdr_list)
                similarity = -sisdrloss(torch.from_numpy(audio1), torch.from_numpy(audio2))
                # print(_id, cluster_id, self.TDoA_options[_id], similarity, check_valid)

                if similarity > SI_SDR_THRESHOLD or check_valid:
                    clusters[final_candidate_id].append(_id)
                    unique = False
                    break
            if unique:
                clusters[_id] = [_id] 
                audio_unique.append(sigs_all_binural[_id])
                # print("add!!!! ", _id, self.TDoA_options[_id], powers[_id])
                
        
        sample_offsets = []
        for cluster_id in clusters:
            # print(cluster_id, self.TDoA_options[cluster_id])
            sample_offsets.append(self.TDoA_options[cluster_id])

        return audio_unique, sample_offsets
