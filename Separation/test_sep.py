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
from tqdm import tqdm
import itertools
import torchaudio.functional as F

sisdrloss = SingleSrcNegSDR('sisdr')


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

def test_blind(searcher, test_set, ROOT_FOLDER, split, ratio, save):

    sisdrloss = SingleSrcNegSDR('sisdr')
    head_size = test_set.head_size
    lang_list = ["fr"]
    map_lists = f"/scr/data_streamspeech/cvss/cvss-c/fr-en/fbank2unit/{split}.tsv"

    save_folders = {}
    save_tsv = {}
    for l in lang_list:
        lang_dir = f"{ROOT_FOLDER}/{l}/{split}/" 
        os.makedirs(lang_dir, exist_ok=True)
        pred_dir =  lang_dir + "pred/"
        gt_dir = lang_dir + "gt/"
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        save_folders[l] = {
            "parent_dir": lang_dir,
            "pred_dir": pred_dir,
            "gt_dir": gt_dir,
        }
        save_tsv[l] = f"{ROOT_FOLDER}/{l}/{split}.tsv"
        
    folder_name = {
        "fr": "/scr/data_streamspeech/cvss/cvss-c/fr-en/fbank2unit",
        "de": "/scr/data_streamspeech_de/cvss/cvss-c/de-en/fbank2unit",
        "es": "/scr/data_streamspeech_es/cvss/cvss-c/es-en/fbank2unit"
    }

    all_lang_mapping = {}
    if save:
        for lang in lang_list:
            mappings = {}
            map_lists = f"{folder_name[lang]}/{split}.tsv"
            with open(map_lists, newline='') as tsvfile:
                reader = csv.DictReader(tsvfile, delimiter='\t')
                for row in reader:
                    # print(row)
                    temp = {
                        "src_text": row["src_text"],
                        "tgt_text": row["tgt_text"]

                    }
                    mappings[row['id']] = temp

            all_lang_mapping[lang] = mappings
    src_lists = {}
    tgt_lists = {}
    wav_lists = {}
    gt_lists = {}
    tsv_dat = {} #[]
    sisdr_list = {}
    angle_err_list = {}
    
    FP = {}
    FN = {}
    TP = {}

    for l in lang_list:
        src_lists[l] = ""
        tgt_lists[l] = ""
        wav_lists[l] = ""
        gt_lists[l] = ""
        tsv_dat[l] = []
        sisdr_list[l] = []
        angle_err_list[l] = []
        FP[l] = 0
        FN[l] = 0
        TP[l] = 0


    try:
        # for idx in range(0, len(test_set)):
        for idx in tqdm(range(0, len(test_set) )):
            if np.random.rand() > ratio:
                continue
            print("-"*10, idx)
            inputs, targets = test_set[idx]
            mix = inputs["mixture"]
            target = targets["target"]
            angle_gt = targets["angle_gt"]
            TDoA_gt = targets["TDoA_gt"]
            fnames = targets["names"]
            langs = targets["langs"]    

            audio_unique, sample_offsets = searcher.process_search(mix)
            audio_unique = [torch.from_numpy(az) for az in audio_unique]
            # print("sample pred = ", sample_offsets)
            # for ai, a in enumerate(audio_unique):
            #     save_mps(a, f"./debug/out{idx}_{ai}.mp3")

            TDoA_matched  = []
            angle_matched = []
            wav_matcched= [] 
            fname_matched = []
            gt_matched = []
            angle_gt_matched = []
            used_id = []
            langs_matched = []
            # print(TDoA_gt, sample_offsets, len(sample_offsets) )
            # if 1:
            if len(audio_unique) > 0:
                best_perm = find_best_permutation(target, audio_unique, TDoA_gt, sample_offsets)
                for out_i, gt_i in best_perm:                        
                    TDoA_matched.append(sample_offsets[out_i])

                    angle1 = conver_TDoA2Az(sample_offsets[out_i], head_size = head_size, sr=16000)
                    angle_matched.append(angle1)

                    angle0 = angle_gt[gt_i]
                    if angle_gt[gt_i] > 90:
                        angle0 = 180 - angle_gt[gt_i]
                    elif angle_gt[gt_i] < -90:
                        angle0 = -180 - angle_gt[gt_i]
                    else:
                        angle0 = angle_gt[gt_i]

                    angle_gt_matched.append(angle0)


                    wav_matcched.append(audio_unique[out_i])
                    gt_matched.append(target[gt_i][0:1])


                    fname_matched.append(fnames[gt_i])
                    langs_matched.append(langs[gt_i])
                    
            lang = langs[0]

            TP[lang] += len(TDoA_matched)

            if len(TDoA_gt) - len(TDoA_matched) > 0:
                print("False negative=", len(TDoA_gt) - len(TDoA_matched))
                FN[lang] += len(TDoA_gt) - len(TDoA_matched)
            elif len(sample_offsets) - len(TDoA_matched) > 0:
                print("False positve=", len(sample_offsets) - len(TDoA_matched))
                FP[lang] += len(sample_offsets) - len(TDoA_matched)

            for j, au in enumerate(wav_matcched):
                name = fname_matched[j]
                lang = langs_matched[j]

                angle0 = angle_gt_matched[j]
                angle1 = angle_matched[j]
                # print(lang, au.shape, gt_matched[j].shape)
                sisdr_in = sisdrloss(mix[0:1], gt_matched[j][0:1])

                sisdr = sisdrloss(au[0:1], gt_matched[j][0:1])
                angle_err = np.abs(angle0 - angle1)
                # sisdr2 = sisdrloss(au[1:2], gt_matched[j][1:2])
                
                print("sisdr = ", j, name, lang, -sisdr_in, -sisdr)
                print("angle diff", angle_err, angle0, angle1)
                # raise KeyboardInterrupt
                sisdro_value = -torch.mean(sisdr).item()
                sisdri_value = -torch.mean(sisdr_in).item()
                sisdr_list[lang].append(sisdro_value - sisdri_value)
                angle_err_list[lang].append(angle_err)
                mappings = all_lang_mapping[lang]
                ### save file
                if name in mappings.keys():

                    src_text = mappings[name]["src_text"]
                    tgt_text = mappings[name]["tgt_text"]
                    src_lists[lang]  += (src_text + '\n')
                    tgt_lists[lang] += (tgt_text + '\n')

                    pred_file = save_folders[lang]["pred_dir"] + f"s{idx}_{name}.mp3"
                    gt_file = save_folders[lang]["gt_dir"] + f"s{idx}_{name}.mp3"

                    wav_lists[lang] +=  (pred_file + '\n')
                    gt_lists[lang] += (gt_file + '\n')
                    save_mps(au, pred_file)
                    save_mps(gt_matched[j], gt_file)
                    # utils.write_audio_file(pred_file, au, 16000)
                    # utils.write_audio_file(gt_file, gt_matched[j], 16000)
                    dat = {}
                    dat["id"] = name
                    dat["sep_id"] = idx 
                    dat["file"] = pred_file
                    dat["src_text"] = src_text
                    dat["tgt_text"] = tgt_text
                    dat["sisdrin"] = sisdri_value
                    dat["sisdro"] = sisdro_value
                    dat["angle_err"] = angle_err
                    tsv_dat[lang].append(dat)
                else:
                    print("Warining!!!!, not found in valid filename", name)
            # utils.write_audio_file(os.path.join(sample_dir, f"prompt{i}.wav"), inputs["prompt"], 24000)
            # utils.write_audio_file(os.path.join(("./debug", f"gt.wav"), gt, 24000)
    finally:
        

        for lang in lang_list: ### save for each lang
            print(f"language {lang}........")
            print("average sisdr = ", np.mean(sisdr_list[lang]))
            print("average angle error = ", np.mean(angle_err_list[lang]))
            print("precision = ", TP[lang]/(TP[lang] + FN[lang]), "recall = ", TP[lang]/(TP[lang] + FP[lang]))
            with open(save_folders[lang]["parent_dir"] + 'wav_list.txt', 'w', encoding='utf-8') as file:
                # Write the text to the file
                file.write(wav_lists[lang])
            file.close()
            with open(save_folders[lang]["parent_dir"]  + 'gt_list.txt', 'w', encoding='utf-8') as file:
                # Write the text to the file
                file.write(gt_lists[lang])
            file.close()
            with open(save_folders[lang]["parent_dir"] + 'source.txt', 'w', encoding='utf-8') as file:
                # Write the text to the file
                file.write(src_lists[lang] )
            file.close()
            with open(save_folders[lang]["parent_dir"] + 'target.txt', 'w', encoding='utf-8') as file:
                # Write the text to the file
                file.write(tgt_lists[lang] )
            file.close()

            headers = []
            for n in tsv_dat[lang][0].keys():
                headers.append(n)

            with open(save_tsv[lang], 'w', newline='', encoding='utf-8') as tsvfile:
                # Create a DictWriter object, passing the file object and the fieldnames (headers)
                writer = csv.DictWriter(tsvfile, fieldnames=headers, delimiter='\t')
                
                # Write the header (column titles) first
                writer.writeheader()
                
                # Write the rows (data)
                writer.writerows(tsv_dat[lang])



if __name__ == "__main__":
    utils.seed_all(10)
    parser = argparse.ArgumentParser()

    parser.add_argument('run_dir',
                        type=str,
                        help='Path to model run')
    parser.add_argument('--use_noise',
                        action='store_true',
                        help='Whether to use cuda')
    parser.add_argument('--sr',
                    type=int,
                    default=16000,
                    help='Project sampling rate')           
    parser.add_argument('--ratio',
                    type=float,
                    default=1,
                    help='ratio of datast to run sep')      
    parser.add_argument('--save',
                    action='store_true',
                    help='whether save')           

    split = "test"
    args = parser.parse_args()
    device = 'cuda' #if args.use_cuda else 'cpu'
    test_set = InferDataset(input_dir = f"/scr/data_blind_separation_noise_alllang/covost2/{split}/", split =split, BG_prob = args.use_noise)  # args.use_noise
    if args.use_noise:
        ROOT_FOLDER = "/scr/blind_sep_alllang_noise"
    else:
        ROOT_FOLDER = "/scr/blind_sep_alllang"

    model = utils.load_torch_pretrained(args.run_dir, model_epoch = "best").model

    model = model.to(device)


    searcher = Binural_Search(args.sr, model, device)

    os.makedirs(ROOT_FOLDER, exist_ok=True)
    test_blind(searcher, test_set, ROOT_FOLDER, split, args.ratio, args.save)
