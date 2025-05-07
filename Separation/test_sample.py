import os 
import src.utils as utils
from Blind_Sep.Binural_Search import Binural_Search
import argparse
import numpy as np 
from asteroid.losses.sdr import SingleSrcNegSDR
import math
import whisper 
from torchaudio.functional import resample
from stable_whisper import modify_model
import json
from src.datasets.dataset_mixing import InferDataset, conver_TDoA2Az, conver_Az2TDoA
import torch
import csv
import io
import soundfile as sf
from pydub import AudioSegment
import re 
import itertools
import torchaudio.functional as F
import librosa
from tqdm import tqdm
# from silero_vad import load_silero_vad, get_speech_timestamps
import torchaudio.transforms as T
import torchaudio
sisdrloss = SingleSrcNegSDR('sisdr')

### mormalize and resample from 16 to 48
def norm_resample(waveform):
    # waveform = torch.from_numpy(waveform)
    resampler = T.Resample(orig_freq=16000, new_freq=48000)

    max_val = torch.max(torch.abs(waveform)) * 1.2

    # Normalize the waveform by the absolute maximum value
    if max_val > 0:
        waveform = waveform / max_val
    else:
        waveform = waveform

    # Apply the resample transform
    upsampled_waveform = resampler(waveform)

    return upsampled_waveform.numpy()


def find_best_permutation(wav_gt, wav_pred , pos_gt, pos_pred, acceptable_range=2, accept_sisdr=-8):
    num_gt = len(pos_gt)
    num_pred = len(pos_pred)
    n = max(num_gt, num_pred)

    neg_sisdr_matrix = np.ones((n, n))*10000
    dis_matrix = np.ones((n, n))*10000

    for i in range(num_gt):
        for j in range(num_pred):
            dis_matrix[i, j] = np.abs(pos_gt[i] - pos_pred[j])
            neg_sisdr_matrix[i, j] = sisdrloss(wav_pred[j][0:1], wav_gt[i][0:1])

    best_perm = None
    best_inliers = -1
    best_err = 10000

    for perm in itertools.permutations(range(n)):
        curr_inliers = 0
        loss_err = []
        paired = []
        for idx1, idx2  in enumerate(perm):
            neg_sisnr_err = neg_sisdr_matrix[idx1, idx2]
            dis_err = dis_matrix[idx1, idx2]
            ### best matching depends on both sample shift error and SISDR error
            loss =  neg_sisnr_err + dis_err
            if dis_err < acceptable_range and neg_sisnr_err < -accept_sisdr:
                curr_inliers += 1
                loss_err.append(loss)
                paired.append((idx2, idx1))  ### idx2 for output id, idx1 for ground-truth id
        if len(loss_err) > 0:
            curr_err = np.mean(loss_err) 
        else:
            curr_err = np.inf 
        if (curr_inliers > best_inliers) or (curr_inliers == best_inliers and curr_err < best_err):
            best_inliers = curr_inliers
            best_perm = paired
            best_err = curr_err
    # print("best_perm: ", best_perm)
    return best_perm

def save_mps(waveform, dst_file):
    wav = F.resample(waveform, 16000, 48000)
    wav = wav.numpy().T
    # Write the NumPy array to the BytesIO object in WAV format
    wav_io = io.BytesIO()
    sf.write(wav_io, wav, 48000, format='WAV')
    # Seek to the start of the BytesIO object
    wav_io.seek(0)
    # Create an AudioSegment from the WAV data in memory
    audio_segment = AudioSegment.from_wav(wav_io)

    # Export the AudioSegment as MP3
    audio_segment.export(dst_file, format='mp3')



if __name__ == "__main__":
    utils.seed_all(10)
    parser = argparse.ArgumentParser()


    parser.add_argument('--run_dir',
                        type=str,
                        help='Path to model run')

    parser.add_argument('--sr',
                    type=int,
                    default=16000,
                    help='Project sampling rate')           


    args = parser.parse_args()

    output_folder = "/gscratch/intelligentsystems/tuochao/offical_git/Spatial-Speech-Translation/data_example/"

    data_folder = [
        "/gscratch/intelligentsystems/tuochao/offical_git/Spatial-Speech-Translation/data_example/example1",
        "/gscratch/intelligentsystems/tuochao/offical_git/Spatial-Speech-Translation/data_example/example2",
        "/gscratch/intelligentsystems/tuochao/offical_git/Spatial-Speech-Translation/data_example/example3",
        "/gscratch/intelligentsystems/tuochao/offical_git/Spatial-Speech-Translation/data_example/023",
        "/gscratch/intelligentsystems/tuochao/offical_git/Spatial-Speech-Translation/data_example/049",
        "/gscratch/intelligentsystems/tuochao/offical_git/Spatial-Speech-Translation/data_example/014",
    ]

    split = "test"
    head_size =  0.18 # head size
    ### load model
    device = 'cuda' #if args.use_cuda else 'cpu'
    model = utils.load_torch_pretrained(args.run_dir, model_epoch = "best").model
    model = model.to(device)
    searcher = Binural_Search(args.sr, model, device)


    wav_lists = []
    tgt_lists = []
    for idx in tqdm(range(0, len(data_folder) )):
        save_folder = data_folder[idx]

        with open(os.path.join(save_folder, 'metadata.json'), 'r') as f:
            datas = json.load(f)
        
        fnames = []
        target = []
        angle_gt = []
        TDoA_gt = []
        gt_texts = []

        for dat in datas:
            name = dat['name']
            fnames.append(name)
            angle = dat['angle']
            angle_gt.append(angle)
            TDoA_gt.append(conver_Az2TDoA(angle, head_size, 16000))
            gt_texts.append(dat['tgt_text'])

            y, sr = librosa.load(os.path.join(save_folder, f"{name}.wav"), sr=16000) 
            target.append(torch.from_numpy(y).float().unsqueeze(0))

        mix, sample_rate = torchaudio.load(os.path.join(save_folder, "mixture.wav"))
        assert(sample_rate == 16000)
        audio_unique, sample_offsets = searcher.process_search(mix)
        audio_unique = [torch.from_numpy(az) for az in audio_unique]
        best_perm = find_best_permutation(target, audio_unique, TDoA_gt, sample_offsets)

        TDoA_matched  = []
        angle_matched = []
        wav_matcched= [] 
        fname_matched = []
        gt_matched = []
        angle_gt_matched = []
        gt_text_matched = []

        for out_i, gt_i in best_perm:   
            TDoA_matched.append(sample_offsets[out_i])
            angle1 = conver_TDoA2Az(sample_offsets[out_i], head_size = head_size, sr=16000)
            angle_matched.append(angle1)

            tgt_txt = gt_texts[gt_i]
            angle0 = angle_gt[gt_i]
            if angle_gt[gt_i] > 90:
                angle0 = 180 - angle_gt[gt_i]
            elif angle_gt[gt_i] < -90:
                angle0 = -180 - angle_gt[gt_i]
            else:
                angle0 = angle_gt[gt_i]

            angle_gt_matched.append(angle0)
            gt_text_matched.append(tgt_txt)

            wav_matcched.append(audio_unique[out_i])
            gt_matched.append(target[gt_i][0:1])
            fname_matched.append(fnames[gt_i])

        data_json = []

        for j, au in enumerate(wav_matcched):
            name = fname_matched[j]

            angle0 = angle_gt_matched[j]
            angle1 = angle_matched[j]
            # print(lang, au.shape, gt_matched[j].shape)
            sisdr_in = sisdrloss(mix[0:1], gt_matched[j][0:1])

            sisdr = sisdrloss(au[0:1], gt_matched[j][0:1])
            angle_err = np.abs(angle0 - angle1)
            # sisdr2 = sisdrloss(au[1:2], gt_matched[j][1:2])
            
            print("input sisdr = ", -sisdr_in, "output sisdr = ", -sisdr)
            print("angle diff", angle_err)

            save_mps(au, os.path.join(save_folder, f"sep_{name}.wav"))
            wav_lists.append(os.path.join(save_folder, f"sep_{name}.wav"))
            tgt_text = gt_text_matched[j]
            tgt_lists.append(tgt_text)


    with open(output_folder + 'wav_list.txt', 'w', encoding='utf-8') as file:
        # Write the text to the file
        for line in wav_lists:
            file.write(line + "\n")
    
    with open(output_folder + 'gt_list.txt', 'w', encoding='utf-8') as file:
        # Write the text to the file
        for line in tgt_lists:
            file.write(line + "\n")

    file.close()

