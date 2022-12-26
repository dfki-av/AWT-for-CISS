#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=0,1
NB_GPU=2

DATA_ROOT=/ds-av/public_datasets/cityscapes/raw/

DATASET=cityscapes
TASK=10-1
NAME=MiB
METHOD=MiB
OPTIONS="--checkpoint checkpoints/step/lr_0.02/ --att 25 --mask_att"

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"
RESULTSFILE=results/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"

# If you already trained the model for the first step, you can re-use those weights
# in order to skip this initial step --> faster iteration on your model
# Set this variable with the weights path
MODEL0=checkpoints/step/att_top_25/10-1-cityscapes_MiB_0.pth

# Then, for the first step, append those options:
# --ckpt ${FIRSTMODEL} --test
# And for the second step, this option:
# --step_ckpt ${FIRSTMODEL}

BATCH_SIZE=12
INITIAL_EPOCHS=30
EPOCHS=30

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 0 --lr 0.02 --epochs ${INITIAL_EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} #--ckpt ${MODEL0} --test
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=1 attribute.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr 0.001 --method ${METHOD} --opt_level O2 ${OPTIONS} #--step_ckpt ${MODEL0}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} #--step_ckpt ${MODEL0} #--test
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=1 attribute.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 2 --lr 0.001 --method ${METHOD} --opt_level O2 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 2 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} #--ckpt ${MODEL2} --test
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=1 attribute.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 3 --lr 0.001 --method ${METHOD} --opt_level O2 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 3 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} #--ckpt ${MODEL3} --test
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=1 attribute.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 4 --lr 0.001 --method ${METHOD} --opt_level O2 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 4 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} #--ckpt ${MODEL4} --test
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=1 attribute.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 5 --lr 0.001 --method ${METHOD} --opt_level O2 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 5 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} #--ckpt ${MODEL5} --test
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=1 attribute.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 6 --lr 0.001 --method ${METHOD} --opt_level O2 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 6 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} #--ckpt ${MODEL6} --test
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=1 attribute.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 7 --lr 0.001 --method ${METHOD} --opt_level O2 ${OPTIONS} --step_ckpt ${MODEL6}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 7 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} #--step_ckpt ${MODEL6} #--test
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=1 attribute.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 8 --lr 0.001 --method ${METHOD} --opt_level O2 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 8 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} #--ckpt ${MODEL8} --test
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=1 attribute.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 9 --lr 0.001 --method ${METHOD} --opt_level O2 ${OPTIONS}
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 9 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS} #--ckpt ${MODEL9} --test
python3 average_csv.py ${RESULTSFILE} ${TASK}

echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"
