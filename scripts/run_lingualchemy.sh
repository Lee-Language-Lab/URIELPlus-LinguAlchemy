#!/bin/bash

EPOCHES=10
# Update these paths to reflect where you want the output and evaluation directories to be on your Windows machine
OUT_DIR="ablation"
EVAL_DIR="outputs"
SCALE=10
VECTOR=syntax_knn_syntax_average_geo

# Set the PYTHONPATH to include the src directory
export PYTHONPATH=$(pwd)/src

# Loop over model names and scales
for MODEL_NAME in bert-base-multilingual-cased xlm-roberta-base;
do
    for SCALE in 1 10 25 50 100 dynamiclearn dynamicscale;
    do
        # Run the Python module with the specified parameters
        CUDA_VISIBLE_DEVICES=4,5 python -m src.lingualchemy \
        --model_name ${MODEL_NAME} --epochs ${EPOCHES}  \
        --out_path ${OUT_DIR}/massive/${MODEL_NAME}/scale${SCALE}_${VECTOR} \
        --vector ${VECTOR} --scale ${SCALE} --eval_path ${EVAL_DIR}/massive/${MODEL_NAME}_scale${SCALE}
    done
done