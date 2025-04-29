export CUDA_VISIBLE_DEVICES=0

ROOT=/gscratch/intelligentsystems/tuochao/StreamSpeech
DATA_ROOT=/scr/data_streamspeech/cvss/cvss-c

LANG=fr
file=/gscratch/intelligentsystems/tuochao/StreamSpeech/checkpoints/streamspeech.simul-s2st.fr-en/checkpoint_27_43000.pt
output_dir=$ROOT/res/streamspeech.simultaneous.${LANG}-en/simul-s2tt

chunk_size=960

PYTHONPATH=$ROOT/fairseq simuleval --data-bin ${DATA_ROOT}/${LANG}-en/fbank2unit \
    --user-dir ${ROOT}/researches/ctc_unity  --agent-dir ${ROOT}/agent \
    --source ${ROOT}/example/wav_list.txt --target ${ROOT}/example/target.txt \
    --model-path $file \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --agent $ROOT/agent/binural_speech_to_text.s2tt.streamspeech.agent.py\
    --output $output_dir/chunk_size=$chunk_size \
    --source-segment-size $chunk_size \
    --quality-metrics BLEU --latency-metrics AL AP DAL StartOffset EndOffset LAAL ATD NumChunks RTF \
    --device gpu --computation-aware 