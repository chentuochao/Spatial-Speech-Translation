export CUDA_VISIBLE_DEVICES=0,1
export SLURM_NTASKS=1
export PYTHONPATH=$PYTHONPATH:/gscratch/intelligentsystems/tuochao/Spatial-StreamSpeech/StreamSpeech/fairseq
export PYTHONPATH=$PYTHONPATH:/gscratch/intelligentsystems/tuochao/Spatial-StreamSpeech/StreamSpeech/SimulEval

. /mmfs1/gscratch/cse/tuochao/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/intelligentsystems/tuochao/conda/StreamSpeech2

ROOT=/gscratch/intelligentsystems/tuochao/Spatial-StreamSpeech/StreamSpeech
DATA_ROOT=/scr/data_streamspeech/cvss/cvss-c
PRETRAIN_ROOT=$ROOT/pretrain_models
VOCODER_CKPT=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000
VOCODER_CFG=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json

LANG=fr
# file=/gscratch/intelligentsystems/shared_runs/translation/simul-s2st.singlechannel-fr-en-seamless/checkpoint_best.pt # 17.18
# file=/gscratch/intelligentsystems/shared_runs/translation/simul-s2st.singlechannel-fr-en-finetune-sep-mix-changed-dict/checkpoint_best.pt # 
file=/gscratch/intelligentsystems/shared_runs/translation/simul-s2st.singlechannel-fr-en-seamless-finetune-noise/checkpoint.best_loss_8.8347.pt # 2.43 # 17.41
# file=/gscratch/intelligentsystems/shared_runs/translation_revision/simul-s2st.singlechannel-fr-en-seamless-finetune-from-noise/checkpoint_best.pt

# indoor_result3 old model no dur pred, 3
# indoor_result2 old model dur pred , 3
# indoor_result4 new model no dur pred, 3

output_dir=$ROOT/official_res/streamspeech.simultaneous.${LANG}-en/demo_output_seamless
# SRC_file=/gscratch/intelligentsystems/qirui/playground/translation/realworld/fr_wav_list.txt
# TGT_file=/gscratch/intelligentsystems/qirui/playground/translation/realworld/fr_target.txt
TGT_file=/mmfs1/gscratch/intelligentsystems/tuochao/offical_git/demo_debug/gt_list.txt
SRC_file=/mmfs1/gscratch/intelligentsystems/tuochao/offical_git/demo_debug/wav_list.txt

chunk_size=960

PYTHONPATH=$ROOT/fairseq simuleval --data-bin ${DATA_ROOT}/${LANG}-en/fbank2unit \
    --user-dir ${ROOT}/researches/ctc_unity --agent-dir ${ROOT}/agent \
    --source $SRC_file\
    --target $TGT_file\
    --model-path $file \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --agent $ROOT/agent/speech_to_speech.seamless.agent.py \
    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG --dur-prediction \
    --output $output_dir \
    --source-segment-size $chunk_size \
    --quality-metrics ASR_BLEU  --target-speech-lang en \
    --device gpu \
    --latency-metrics AL AP DAL StartOffset EndOffset LAAL ATD NumChunks DiscontinuitySum DiscontinuityAve DiscontinuityNum RTF \
    --silero-vad 1 \
    --silence-ignore --seamless-dur 2 \


# To calculate ASR-BLEU w/o silence,
# Another way: You can simply comment out Line 358 to Line 360 of StreamSpeech/SimulEval/simuleval/evaluator/instance.py to prevent silence from being added to the result within SimulEval.

# conda activate /gscratch/intelligentsystems/tuochao/conda/StreamSpeech

# cd $ROOT/asr_bleu_rm_silence
# python compute_asr_bleu.py \
#     --reference_path  $TGT_file \
#     --lang en --audio_dirpath $output_dir/wavs \
#     --reference_format txt \
#     --transcripts_path $output_dir/rm_silence_asr_transcripts.txt \
#     --results_dirpath $output_dir/rm_silence_asr_bleu
# # cd $ROOT