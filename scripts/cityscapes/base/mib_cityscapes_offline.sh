#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=0,1
NB_GPU=2

DATA_ROOT=/ds-av/public_datasets/cityscapes/raw/

DATASET=cityscapes
TASK=offline
NAME=MiB
METHOD=MiB
OPTIONS="--checkpoint checkpoints/step/lr_0.02/base/ --orig_init"

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"
RESULTSFILE=results/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

BATCH_SIZE=12
INITIAL_EPOCHS=30
EPOCHS=30

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.02 --epochs ${INITIAL_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}
python3 average_csv.py ${RESULTSFILE} ${TASK}

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"
