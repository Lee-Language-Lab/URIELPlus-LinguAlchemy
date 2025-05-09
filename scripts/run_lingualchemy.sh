#!/bin/bash

EPOCHES=30
CUR_FILE_PATH=$(realpath "$0")
CUR_FOLDER_PATH=$(dirname "$CUR_FILE_PATH")
ROOT_DIR=$(dirname "$CUR_FOLDER_PATH")
OUT_DIR="$ROOT_DIR/ablation"  # set customised out_path
EVAL_DIR="$ROOT_DIR/outputs"

SCALE=10

for MODEL_NAME in bert-base-multilingual-cased xlm-roberta-base;
do
    for VECTOR in geo.pt syntax_average_geo.pt syntax_average.pt syntax_knn_geo.pt syntax_knn_syntax_average_geo.pt syntax_knn_syntax_average.pt syntax_knn.pt;
    do
        for DATASET in masakhanews_vectors massive_vectors semrel_vectors;
        do
            CUDA_VISIBLE_DEVICES=0,1 python3 -m src.main \
                --model_name ${MODEL_NAME} --epochs ${EPOCHES}  \
                --out_path ${OUT_DIR}/${DATASET}/${MODEL_NAME}/scale${SCALE}_${VECTOR} --dataset ${DATASET} \
                --vector ${VECTOR} --scale ${SCALE} --eval_path ${EVAL_DIR}/${DATASET}/${MODEL_NAME}_scale${SCALE} --wandb_offline
        done
    done
done