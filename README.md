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

## 🏋️‍♂️ Model training

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

### Inference the binuaral model
Inference on synthetic dataset
```
python test_sep.py /gscratch/intelligentsystems/shared_runs/translation/angle_sep_alllang_fdown_ft/
```

Inference on real-world dataset
```
python test_realworld.py fr_q --run_dir /gscratch/intelligentsystems/shared_runs/translation/angle_sep_alllang_fdown_ft/
```


## Translation Module (FR-EN)
### Pretrain the base speech translation model
Download the processed COVOST and CVSS [dataset](https://drive.google.com/file/d/1Y3qQVZdSEj3f_BXn7b9r4TLcybv5S1RR/view?usp=sharing)
```
tar -xvf /gscratch/intelligentsystems/common_datasets/translation/data_stream_1channel_processed.tar.gz -C DATA_FOLDER
./official_Script/train.simul-s2s-fr-en.sh
```

### Separation-aware finetune on base model
Download the processed COVOST and CVSS  [dataset](https://drive.google.com/file/d/1Lpsg3wo6l5vubJ8KDiANId0bW3Xc7MSP/view?usp=sharing) with imperfect Separation
```
tar -xvf /gscratch/intelligentsystems/common_datasets/translation/mixing_dataset_fr_processed.tar -C DATA_FOLDER
./official_Script/train.simul-s2s-fr-en-noise.sh
```

### Train the expressive speech unit generator
Download the processed COVOST and CVSS [dataset](https://drive.google.com/file/d/1PRexFNCZxhxT8lw8v7DVaIab-YloJjGr/view?usp=sharing) with expressive units
```
tar -xvf /gscratch/intelligentsystems/common_datasets/translation/data_stream_1channel_seamless.tar -C DATA_FOLDER
./official_Script/train.simul-s2s-finetune-noise-expressive.sh
```

### Inference
```
tar -xvf /gscratch/intelligentsystems/common_datasets/translation/blind_sep_alllang_small_dev.tar -C DATA_FOLDER
./official_Script/simuleval.simul-s2st.sh
```


## 🙏 Acknowledgement
We first thank the all authors and contributors of the open-source codes and repos used in our projects. Our streaming speech-to-text module is based on [StreamSpeech](https://github.com/ictnlp/StreamSpeech/tree/main), our expressive text-to-speech module is based on [Seamless Communication](https://github.com/facebookresearch/seamless_communication/tree/main). Our seperation architecture is based on [TF-GridNet](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnetv2_separator.py).