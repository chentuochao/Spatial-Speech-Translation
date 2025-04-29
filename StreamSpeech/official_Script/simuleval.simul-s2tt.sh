export CUDA_VISIBLE_DEVICES=0
export SLURM_NTASKS=1
export PYTHONPATH=$PYTHONPATH:/gscratch/intelligentsystems/tuochao/Spatial-StreamSpeech/StreamSpeech/fairseq
export PYTHONPATH=$PYTHONPATH:/gscratch/intelligentsystems/tuochao/Spatial-StreamSpeech/StreamSpeech/SimulEval


ROOT=/gscratch/intelligentsystems/tuochao/Spatial-StreamSpeech/StreamSpeech
DATA_ROOT=/scr/data_streamspeech/cvss/cvss-c

LANG=fr

# file=/gscratch/intelligentsystems/shared_runs/translation_demo/simul-s2t.singlechannel-${LANG}-en/checkpoint_best.pt
file=/gscratch/intelligentsystems/shared_runs/translation_demo/simul-s2st.singlechannel-${LANG}-en-finetune-noise/checkpoint_best.pt
# file=/gscratch/intelligentsystems/shared_runs/translation/simul-s2st.singlechannel-fr-fix2/checkpoint_best.pt
# file=/gscratch/intelligentsystems/shared_runs/translation/simul-s2st.singlechannel-fr-en-finetune-sep-mix-changed-dict/checkpoint_best.pt

# output_dir=$ROOT/official_res/streamspeech.simultaneous.${LANG}-en/simul-s2tt
output_dir=/gscratch/intelligentsystems/tuochao/offical_git/demo_debug/s2tt_res
chunk_size=960
# TGT_file=/scr/blind_sep_alllang_small/${LANG}/test/target.txt 
# SRC_file=/scr/blind_sep_alllang_small/${LANG}/test/gt_list.txt
# SRC_file=/scr/blind_sep_alllang_small/${LANG}/test/wav_list.txt
SRC_file=/mmfs1/gscratch/intelligentsystems/tuochao/offical_git/demo_debug/wav_list.txt
TGT_file=/mmfs1/gscratch/intelligentsystems/tuochao/offical_git/demo_debug/gt_list.txt

simuleval --data-bin ${DATA_ROOT}/${LANG}-en/fbank2unit \
    --user-dir ${ROOT}/researches/ctc_unity  --agent-dir ${ROOT}/agent \
    --source $SRC_file\
    --target $TGT_file\
    --model-path $file \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --agent $ROOT/agent/speech_to_text.s2tt.streamspeech.agent.py\
    --output $output_dir/chunk_size=$chunk_size \
    --source-segment-size $chunk_size \
    --quality-metrics BLEU_INDIVIDUAL --latency-metrics AL AP DAL StartOffset EndOffset LAAL ATD NumChunks RTF \
    --device gpu 
