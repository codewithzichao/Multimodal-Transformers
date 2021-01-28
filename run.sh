#!/bin/bash

base_path="/Users/codewithzichao/Desktop/competitions/meme_EACL2021"
model_name="xlm-roberta"

cd ${base_path}

python3 -u main.py \
  --base_path ${base_path} \
  --model_name ${model_name} \
  --epochs 50 \
  --batch_size 8 \
  --max_norm 0.25 \
  --fold_num 5

echo "finished, have a good day!"
