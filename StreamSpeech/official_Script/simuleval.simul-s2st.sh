export CUDA_VISIBLE_DEVICES=0
export SLURM_NTASKS=1
export PYTHONPATH=$PYTHONPATH:/gscratch/intelligentsystems/tuochao/Spatial-StreamSpeech/StreamSpeech/fairseq
export PYTHONPATH=$PYTHONPATH:/gscratch/intelligentsystems/tuochao/Spatial-StreamSpeech/StreamSpeech/SimulEval


ROOT=/gscratch/intelligentsystems/tuochao/Spatial-StreamSpeech/StreamSpeech
DATA_ROOT=/scr/data_streamspeech/cvss/cvss-c
PRETRAIN_ROOT=$ROOT/pretrain_models
VOCODER_CKPT=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000
VOCODER_CFG=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json

LANG=fr
# file=/gscratch/intelligentsystems/shared_runs/translation_demo/simul-s2st.singlechannel-fr-en-finetune-noise-old/checkpoint_best.pt
file=/gscratch/intelligentsystems/shared_runs/translation/simul-s2st.singlechannel-fr-fix2/checkpoint_best.pt
# file=/gscratch/intelligentsystems/shared_runs/translation/simul-s2st.singlechannel-fr-fix2/checkpoint_best.pt
# file=/gscratch/intelligentsystems/shared_runs/translation/simul-s2st.singlechannel-fr-en-finetune-sep-mix-changed-dict/checkpoint_best.pt

output_dir=$ROOT/official_res/streamspeech.simultaneous.${LANG}-en/demo_mix
# TGT_file=/scr/blind_sep_alllang_small/${LANG}/test/target.txt 
# SRC_file=/scr/blind_sep_alllang_small/${LANG}/test/wav_list.txt
TGT_file=/mmfs1/gscratch/intelligentsystems/tuochao/offical_git/demo_debug/mix_gt.txt
SRC_file=/mmfs1/gscratch/intelligentsystems/tuochao/offical_git/demo_debug/mix_list.txt

chunk_size=960

PYTHONPATH=$ROOT/fairseq simuleval --data-bin ${DATA_ROOT}/${LANG}-en/fbank2unit \
    --user-dir ${ROOT}/researches/ctc_unity --agent-dir ${ROOT}/agent \
    --source $SRC_file  \
    --target  $TGT_file \
    --model-path $file \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --agent $ROOT/agent/speech_to_speech.streamspeech.agent.py \
    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG --dur-prediction \
    --output $output_dir/chunk_size=$chunk_size \
    --source-segment-size $chunk_size \
    --quality-metrics ASR_BLEU  --target-speech-lang en \
    --device gpu --computation-aware \
    --latency-metrics AL  AP DAL StartOffset EndOffset LAAL ATD NumChunks DiscontinuitySum DiscontinuityAve DiscontinuityNum RTF \
    --start-index 0  --end-index 2 \
    --silence-ignore \
    # --extra-output-dir $output_dir/chunk_size=$chunk_size

# # To calculate ASR-BLEU w/o silence,
# # Another way: You can simply comment out Line 358 to Line 360 of StreamSpeech/SimulEval/simuleval/evaluator/instance.py to prevent silence from being added to the result within SimulEval.

cd $ROOT/asr_bleu_rm_silence
python compute_asr_bleu.py --reference_path  $TGT_file --lang en --audio_dirpath $output_dir/chunk_size=$chunk_size/wavs --reference_format txt --transcripts_path $output_dir/chunk_size=$chunk_size/rm_silence_asr_transcripts.txt --results_dirpath $output_dir/chunk_size=$chunk_size/rm_silence_asr_bleu
cd $ROOT