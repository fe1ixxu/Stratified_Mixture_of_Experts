#!/bin/bash

#SBATCH --job-name=hmoe15
#SBATCH --output=./logs/sample-%j.out
#SBATCH --error=./logs/sample-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4320
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8
#SBATCH --mem 256G
#SBATCH --constraint=volta32gb
#SBATCH --signal=B:USR1@180
#SBATCH --mail-user haoranxu@fb.com
#SBATCH --mail-type end
#SBATCH --partition=learnaccel,nllb

# module load anaconda3/5.0.1
# module load cudnn/v8.0.3.33-cuda.11.0
# module load cuda/11.0
# module load openmpi/4.1.0/cuda.11.0-gcc.9.3.0
# source /public/apps/anaconda3/2021.05/bin/activate fairseq-moe

trap_handler () {
    echo "Caught signal: " $1
    # SIGTERM must be bypassed
    if [ "$1" = "TERM" ]; then
        echo "bypass sigterm"
    else
        # Submit a new job to the queue
        echo "Requeuing " $SLURM_JOB_ID
        scontrol requeue $SLURM_JOB_ID
    fi
}


# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

LANGS='nso,run,ssw,ind,msa,isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim,eng'
LANG_PAIRS='nso-eng,run-eng,ssw-eng,ind-eng,msa-eng,isl-eng,nob-eng,fao-eng,slv-eng,tgl-eng,cat-eng,glg-eng,fur-eng,ltz-eng,lim-eng,eng-nso,eng-run,eng-ssw,eng-ind,eng-msa,eng-isl,eng-nob,eng-fao,eng-slv,eng-tgl,eng-cat,eng-glg,eng-fur,eng-ltz,eng-lim'

DATA_DIR=../m15_32k/
DATA_BIN=${DATA_DIR}/data_bin/shard000/

SAVE_PATH=${1}
TYPE=${2:-'base'}
EXPERT_NUM=${3:-8}
HMOE_LAYER=${4:-2}
DYNAMIC_LAYER=${5:-'4,4'}
HNORM=${6:-1}
MAX_UPDATES=${7:-100000}
MOE_FREQ=2
SIZE='l'
ARCH=transformer
FREQ=1
MAX_TOKENS=4096

if [ ${TYPE} == 'base' ]; then
    EOM="--moe-eom 0.0 --moe-gate-loss-wt 0.01"
    CMR=""
elif [ ${TYPE} == 'eom' ]; then
    EOM="--moe-eom 0.1 --moe-gate-loss-wt 0.01"
    CMR=""
elif [ ${TYPE} == 'cmr' ]; then
    EOM="--moe-eom 0.0 --moe-gate-loss-wt 0.01"
    CMR="--moe-cmr --cmr-gate-drop 0.1 --cmr-gate-loss-wt 0.01 --cmr-gate-loss-p 0.8"
elif [ ${TYPE} == 'eom-cmr' ]; then
    EOM="--moe-eom 0.0 --moe-gate-loss-wt 0.01"
    CMR="--moe-cmr --cmr-gate-drop 0.1 --cmr-gate-loss-wt 0.01 --cmr-gate-loss-p 0.8"
fi


if [ ${SIZE} == 'l' ]; then
    LAYER=6
    DIM=512
    FFN_DIM=2048
    HEADS=8
elif [ ${SIZE} == 'xl' ]; then
    LAYER=6
    DIM=1024
    FFN_DIM=4096 
    HEADS=16
elif [ ${SIZE} == 'xxl' ]; then
    LAYER=12
    DIM=1024
    FFN_DIM=4096
    HEADS=16
fi

mkdir -p ${SAVE_PATH}
RANDOM_PORT=175$(( $RANDOM % 50 + 1 ))
## Train 
#  srun --job-name baseline_naive --output ${SAVE_PATH}/train.%j --error ${SAVE_PATH}/train.stderr.%j --mail-user haoranxu@fb.com --mail-type end \
#  --nodes=4 --ntasks-per-node=1 --time=4320 --cpus-per-task=10 --gpus-per-node 8 --constraint volta32gb \
#  --open-mode append --unbuffered --cpu-bind=map_ldom:0,0,0,0,1,1,1,1 \
 python train.py  ${DATA_BIN} --arch ${ARCH}  --task translation_multi_simple_epoch \
 --lang-pairs ${LANG_PAIRS} --langs ${LANGS} --sampling-method temperature --sampling-temperature 1 --encoder-langtok tgt --decoder-langtok \
 --encoder-layers ${LAYER} --decoder-layers ${LAYER} --encoder-ffn-embed-dim ${FFN_DIM} --decoder-ffn-embed-dim ${FFN_DIM} \
 --encoder-embed-dim ${DIM} --decoder-embed-dim ${DIM} --encoder-attention-heads ${HEADS} --decoder-attention-heads ${HEADS} --attention-dropout 0.1 --relu-dropout 0.0 \
 --decoder-normalize-before --encoder-normalize-before --share-all-embeddings --max-source-positions 512 --max-target-positions 512 \
 --max-update ${MAX_UPDATES} --update-freq ${FREQ}  --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0008 --stop-min-lr 1e-09 --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 --criterion moe_label_smoothed_cross_entropy_ed \
 --label-smoothing 0.1 --best-checkpoint-metric loss --max-tokens ${MAX_TOKENS}  --validate-interval-updates 500 --save-interval-updates 500 --save-interval 2 \
 --keep-interval-updates 1  --validate-interval 1000  --seed 42 --log-format simple --log-interval 100 \
 --fp16 --optimizer adam --min-params-to-wrap 100000000  --use-local-shard-size \
 --save-dir ${SAVE_PATH}  --skip-invalid-size-inputs-valid-test --memory-efficient-fp16  --ddp-backend fully_sharded 
 
exit
# Evaluate
SRCS='nso,run,ssw,ind,msa,isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim'
tgt=eng
mkdir -p ${SAVE_PATH}/results
replication_count=$[ 32 / ${EXPERT_NUM} ]
for src in ${SRCS//,/ }; do
    echo predict $src to $tgt
    FSRC=${DATA_DIR}/retrieved_data/test.${tgt}-${src}.${src}
    FTGT=${DATA_DIR}/retrieved_data/test.${tgt}-${src}.${tgt}
    FOUT=${SAVE_PATH}/results/predict.${tgt}-${src}.${tgt}

    fairseq-generate ${DATA_BIN} --path $SAVE_PATH/checkpoint_best.pt \
        --langs ${LANGS} \
        --lang-pairs ${LANG_PAIRS} \
        --task translation_multi_simple_epoch \
        --is-moe \
        --sacrebleu \
        --encoder-langtok tgt --decoder-langtok \
        --bpe "sentencepiece" \
        --sentencepiece-model ${DATA_DIR}/vocab_bin/sentencepiece.source.32000.model \
        --source-lang ${src} --target-lang ${tgt} \
        --distributed-world-size 32 --distributed-port ${RANDOM_PORT} \
        --batch-size 25  \
        --model-overrides "{'world_size': 32, 'moe_eval_capacity_token_fraction': 1.0, 'use_moe_pad_mask': False, 'pass_tokens_transformer_layer': False, 'replication_count': ${replication_count}}" \
        --no-progress-bar |\
        tail -n 1 >  $FOUT.bleu
    cat ${FOUT}.bleu
done

TGTS='nso,run,ssw,ind,msa,isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim'
src=eng
for tgt in ${TGTS//,/ }; do
    echo predict $src to $tgt
    FSRC=${DATA_DIR}/retrieved_data/test.${src}-${tgt}.${src}
    FTGT=${DATA_DIR}/retrieved_data/test.${src}-${tgt}.${tgt}
    FOUT=${SAVE_PATH}/results/predict.${src}-${tgt}.${tgt}

    fairseq-generate ${DATA_BIN} --path $SAVE_PATH/checkpoint_best.pt \
        --langs ${LANGS} \
        --lang-pairs ${LANG_PAIRS} \
        --task translation_multi_simple_epoch \
        --is-moe \
        --bpe "sentencepiece" \
        --sacrebleu \
        --encoder-langtok tgt --decoder-langtok \
        --sentencepiece-model ${DATA_DIR}/vocab_bin/sentencepiece.source.32000.model \
        --source-lang ${src} --target-lang ${tgt} \
        --distributed-world-size 32 --distributed-port ${RANDOM_PORT} \
        --batch-size 100 \
        --model-overrides "{'world_size': 32, 'moe_eval_capacity_token_fraction': 1.0, 'use_moe_pad_mask': False, 'pass_tokens_transformer_layer': False, 'replication_count': ${replication_count}}" \
        --no-progress-bar |\
        tail -n 1 >  $FOUT.bleu
    cat ${FOUT}.bleu
done



# Print
SRCS='nso,run,ssw,ind,msa,isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim'
tgt=eng
for src in ${SRCS//,/ }; do
    FOUT=${SAVE_PATH}/results/predict.${tgt}-${src}.${tgt}
    echo -------------------
    echo ${src}-${tgt}
    cat $FOUT.bleu | cut -d ' ' -f 7
done

# Print
TGTS='nso,run,ssw,ind,msa,isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim'
src=eng
for tgt in ${TGTS//,/ }; do
    FOUT=${SAVE_PATH}/results/predict.${src}-${tgt}.${tgt}
    echo -------------------
    echo ${src}-${tgt}
    cat $FOUT.bleu | cut -d ' ' -f 7
done


echo 'eng->xx'
python ./get_m15_mean.py \
    --input ${SAVE_PATH}/results/
    
echo 'xx->eng'
python ./get_m15_mean.py \
    --input ${SAVE_PATH}/results/ \
    --engtgt 1
