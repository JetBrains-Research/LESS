#!/bin/bash

gradient_path=$1
train_file_names=$2
ckpts=$3
checkpoint_weights=$4

validation_gradient_path=$5
target_task_names=$6
target_task_files=$7
output_path=$8

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

python3 -m less.data_selection.matching \
--gradient_path $gradient_path \
--train_file_names $train_file_names \
--ckpts $ckpts \
--checkpoint_weights $checkpoint_weights \
--validation_gradient_path $validation_gradient_path \
--target_task_names $target_task_names \
--target_task_files $target_task_files
--output_path $output_path
