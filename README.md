# Spatial Speech Translation: Translating Across Space With Binaural Hearables
## 🗣️ Spatial-Speech-Translation
The official repo for paper "Spatial Speech Translation: Translating Across Space With Binaural Hearables" in CHI 2025

Video Demo in Youtube:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/zxs5QQgengs/0.jpg)](https://www.youtube.com/watch?v=zxs5QQgengs)

## 💡 Features
* We first enable speech translation under multi-speaker and interference conditions.
* Our simultaneous and expressive speech translation model can run in real-time on Apple silicon.
* First binaural rendering of speech translation can preserve spatial cues from the input to the translated output.


## 📑 Open-source Plan

- [x] Inference code and checkpoints for Fr-En translation
- [x] Training code for Fr-En translation
- [ ] Opensource other language (De, Es) datasets, preprocessing, and checkpoints

## 🔧Quick Start
### Setup Sepearation environment
```
conda create -n sep python=3.8
conda activate sep 
cd Separation
pip install -r requirements.txt
```

### Setup StreamSpeech environment
Create the conda environment
```
conda create -n StreamSpeech python==3.10
conda activate StreamSpeech
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install Cython==3.0.10
pip install numpy
pip install --upgrade pip==23.0 # update the pip
```
Install StreamSpeech, before installing fairseq, make sure you have gcc/11.2.0 module in your linux 
```
cd StreamSpeech
cd fairseq
pip install --editable ./ --no-build-isolation #install fairseq
cd SimulEval
pip install --editable ./ #install SimulEval
pip install editdistance
```
Install other IO essential
```
conda install -c conda-forge sox
conda install conda-forge::ffmpeg
```


## 🤖 Quick try data samples
We provide some binural mixtures as input of under data_example folder.
For each sample, it contains
```
mixture.wav - binural mixtures
metadata.json - the name, angle, source and target text in the mixture
common***.wav - ground-truth individual speech before mixing
```

Download the checkpoints: [separation model](https://drive.google.com/file/d/1d2EfJteg6xRmYPH4g4rjXDw4LkBGJBjo/view?usp=share_link) and [translation checkpoints](https://drive.google.com/drive/folders/1FzSMhGmIYJtt1jaQJGkcRy6HbXmNd_vl?usp=sharing).

Run the Separation framework, (change the data folder in the code). It will save the separated wav and wav_list.txt and gt_list.txt back to the data_example folder.
```
cd Separation
python test_sample.py --run_dir SEPARATION_MODEL_PATH
```
Then run the translation model on the separated french wav, (change the path in the bash scripts)
```
cd StreamSpeech
./official_Script/simuleval.simul-s2st-expressive.sh
```



## 🏋️‍♂️ Model training and Inference

### Train the Joint Separation & Localization model
Prepare spatialized mixing dataset ([Download](https://drive.google.com/file/d/1qTYQvg_LFeodnQEMxEGY4zQ1ZP6McreP/view?usp=sharing))
```
tar -xvf /gscratch/intelligentsystems/common_datasets/translation/data_blind_separation_multilang.tar -C DATA_FOLDER
```

Train step1
```
python src/train.py --config ./config/angle_sep_alllang_small.json --run_dir /gscratch/intelligentsystems/shared_runs/translation/angle_sep_alllang_fdown
```
Train step2
```
python src/train.py --config ./config/angle_sep_alllang_small_ft.json --run_dir /gscratch/intelligentsystems/shared_runs/translation/angle_sep_alllang_fdown_ft
```
You can also use trained [separation model](https://drive.google.com/file/d/1d2EfJteg6xRmYPH4g4rjXDw4LkBGJBjo/view?usp=share_link)


### Inference of separation and localiaztion on noisy mixture
Prepare test set for separation and localiaztion
```
./unzip_testset.sh
```

Inference on synthetic dataset without background noise
```
python test_sep.py /gscratch/intelligentsystems/shared_runs/translation/angle_sep_alllang_fdown_ft/
```

Inference on synthetic dataset with background noise
```
python test_sep.py /gscratch/intelligentsystems/shared_runs/translation/angle_sep_alllang_fdown_ft/ --use_noise
```

### Train Translation Module (FR-EN)
Three-step training recipe, or you can use trained [translation checkpoints](https://drive.google.com/drive/folders/1FzSMhGmIYJtt1jaQJGkcRy6HbXmNd_vl?usp=sharing)

(1) Pretrain the base speech translation model

Download the processed COVOST and CVSS [dataset](https://drive.google.com/file/d/1Y3qQVZdSEj3f_BXn7b9r4TLcybv5S1RR/view?usp=sharing)
```
tar -xvf /gscratch/intelligentsystems/common_datasets/translation/data_stream_1channel_processed.tar.gz -C DATA_FOLDER
./official_Script/train.simul-s2s-fr-en.sh
```

(2) Separation-aware finetune on base model

Download the processed COVOST and CVSS  [dataset](https://drive.google.com/file/d/1Lpsg3wo6l5vubJ8KDiANId0bW3Xc7MSP/view?usp=sharing) with imperfect Separation
```
tar -xvf /gscratch/intelligentsystems/common_datasets/translation/mixing_dataset_fr_processed.tar -C DATA_FOLDER
./official_Script/train.simul-s2s-fr-en-noise.sh
```

(3) Train the expressive speech unit generator

Download the processed COVOST and CVSS [dataset](https://drive.google.com/file/d/1PRexFNCZxhxT8lw8v7DVaIab-YloJjGr/view?usp=sharing) with expressive units
```
tar -xvf /gscratch/intelligentsystems/common_datasets/translation/data_stream_1channel_seamless.tar -C DATA_FOLDER
./official_Script/train.simul-s2s-finetune-noise-expressive.sh
```

### Inference for translation
Run evaluation on the separated audio from the separation and localiaztion without expressive output
```
tar -xvf /gscratch/intelligentsystems/common_datasets/translation/data_stream_1channel_processed.tar.gz -C DATA_FOLDER
tar -xvf /gscratch/intelligentsystems/common_datasets/translation/blind_sep_alllang_small_dev.tar -C DATA_FOLDER
./official_Script/simuleval.simul-s2st.sh
```

Run evaluation on the separated audio from the separation and localiaztion with expressive output
```
tar -xvf /gscratch/intelligentsystems/common_datasets/translation/data_stream_1channel_seamless.tar -C DATA_FOLDER
tar -xvf /gscratch/intelligentsystems/common_datasets/translation/blind_sep_alllang_small_dev.tar -C DATA_FOLDER
./official_Scriptsimuleval.simul-s2st-expressive.sh
```


## 🙏 Acknowledgement
We first thank the all authors and contributors of the open-source codes and repos used in our projects. Our streaming speech-to-text module is based on [StreamSpeech](https://github.com/ictnlp/StreamSpeech/tree/main), our expressive text-to-speech module is based on [Seamless Communication](https://github.com/facebookresearch/seamless_communication/tree/main). Our seperation architecture is based on [TF-GridNet](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnetv2_separator.py).