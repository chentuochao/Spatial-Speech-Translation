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
file=XXX/simul-s2s-${LANG}-en-ftnoise-expressive/checkpoint_best.pt

output_dir=$ROOT/result/streamspeech.simultaneous.${LANG}-en/output_expressive
TGT_file=/scr/blind_sep_alllang_small/${LANG}/test/target.txt 
SRC_file=/scr/blind_sep_alllang_small/${LANG}/test/wav_list.txt

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

conda activate /gscratch/intelligentsystems/tuochao/conda/StreamSpeech
cd $ROOT/asr_bleu_rm_silence
python compute_asr_bleu.py \
    --reference_path  $TGT_file \
    --lang en --audio_dirpath $output_dir/wavs \
    --reference_format txt \
    --transcripts_path $output_dir/rm_silence_asr_transcripts.txt \
    --results_dirpath $output_dir/rm_silence_asr_bleu
cd $ROOT