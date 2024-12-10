# LESS: Selecting Influential Data for Targeted Instruction Tuning

This repo contains the code for our ICML 2024  paper [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/abs/2402.04333). In this work, we propose a data selection method to select influential data to induce a target capability.

## 🔗 Quick Links
- [LESS: Selecting Influential Data for Targeted Instruction Tuning](#less-selecting-influential-data-for-targeted-instruction-tuning)
  - [🔗 Quick Links](#-quick-links)
  - [Install Requirements](#install-requirements)
  - [Data Preparation](#data-preparation)
  - [Data Selection Pipeline](#data-selection-pipeline)
    - [Step 1: Warmup training](#step-1-warmup-training)
    - [Step 2: Building the gradient datastore](#step-2-building-the-gradient-datastore)
    - [Step 3: Selecting data for a task](#step-3-selecting-data-for-a-task)
    - [Step 4: Train with your selected data](#step-4-train-with-your-selected-data)
  - [Evaluation](#evaluation)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)


## Install Requirements
**Step 1**: To get started with this repository, you'll need to follow these installation steps. Before proceeding, make sure you have [Pytorch](https://pytorch.org/get-started/previous-versions/) installed. 
```
pip3 install torch==2.1.2 torchvision torchaudio
```

**Step 2**: Then install the rest of the required packages:
```
cd LESS
pip install -r requirement.txt
or 
poetry install
```

**Step 3**: Finally, install the `less` package in editable mode to make it accessible for your development environment:
```
pip install -e .
```


## Data Preparation
We follow the [open-instruct](https://github.com/allenai/open-instruct?tab=readme-ov-file#dataset-preparation) repo to prepare four instruction tuning datasets. In our project, we utilize a combination of four training datasets: Flan v2, COT, Dolly, and Open Assistant. For the purposes of evaluation, we employ three additional datasets: MMLU, Tydiqa, and BBH. A processed version of these files are available [here](https://huggingface.co/datasets/princeton-nlp/less_data).

## Data Selection Pipeline

### Step 1: Warmup training
To enhance downstream performance from data selection, it's crucial to start with a warmup training step. This involves selecting a small portion of your entire dataset to train using the LoRA method. Follow these steps for effective warmup training:

```bash 
python3 -m less/scripts/train/warmup_lora_train --train_file <str> --model_path <str>
```
NB: there are more optional arguments that you can use to alter the training process. Please refer to the script for more details.
You can also set `--percentage` to specify the percentage of data to train on (default is 0.05) and `--data_seed` to specify the seed for data selection (default is 3).
The checkpoint will be saved in the `out` directory.

### Step 2: Building the gradient datastore
Once the initial warmup training stage is completed, we will collect gradients for the entire training dataset. For each checkpoint, our goal is to obtain the gradients of all the training data that we would like to select from. An example script is shown below.

```bash
python3 -m less/scripts/get_info/grad/get_train_lora_grads \
  --train_data_name <str> \
  --train_file <str> \
  --model_path <str> \
  --ckpts <str> \
  --dims <int>
```
Ideally, you would aim to create a datastore that encompasses a gradient of all the checkpoints and training data from which you wish to choose.  
`train_data_name` is the name of the training data, which will be used to store the gradients, it should be comprehensive for you to easily distignuish between different experiments.  
`train_file` is the path to the training file.  
`model_path` is the path to the model in the `out` directory, e.g. `llama2-7b-p0.05-lora-seed3`.  
`ckpts` is the list of checkpoints to compute gradients for, e.g. `105 211 317 420`. The paper recommends using all four checkpoints.  
`dims` is the dimension of projection, default is 8192.

The gradients will be saved in the `grads` directory.

### Step 3: Selecting data for a task
To select data for a particular downstream task, it's necessary to first prepare data specific to that task, using the same instruction-tuning prompt format as was employed during training. We have set up data loading modules for three evaluation datasets featured in our work: BBH, TydiQA, and MMLU. If you're interested in data selection for additional tasks, you can expand the [`less/data_selection/get_validation_dataset.py`](less/data_selection/get_validation_dataset.py) script to accommodate those tasks. Similar to obtaining gradients for training data, run the following script. The primary difference is that this process will yield SGD gradients for the validation data, following the formulation of the influence estimation. 

You should gain the gradients of the validation data for all the checkpoints you used for building the gradient datastore in the previous step. 

```bash
python3 -m less/scripts/get_info/grad/get_eval_lora_grads \
  --task <str> \
  --data_dir <str> \
  --val_task_load_method <str> \
  --model_path <str> \
  --ckpts <str> \
  --dims <int>
```
`task` is the name of the task, which will be used to store the gradients.  
`data_dir` is the path to the data directory. If you are using one of the predifined datasets ("bbh", "tydiqa", "mmlu"), this should point to the data directory. If you are using your own custom dataset, this should be a full path to a JSONL file or a HF repo name.  
`val_task_load_method` is the method to load the validation data, can be `hf`, `local_hf`, `local_json`. You should specify this if you are using your own custom dataset. Default is `None`, then it's assumned that you are using the predifined datasets.  
`model_path` is the path to the model in the `out` directory, e.g. `llama2-7b-p0.05-lora-seed3`.  
`ckpts` is the list of checkpoints to compute gradients for, e.g. `'105 211 317 420'`.  
`dims` is the dimension of projection, default is 8192.

The gradients will be saved in the `grads` directory.

After obtaining the gradients for the validation data, we can then select data for the task. The following script will calculate the influence score for each training data point, and select the top-k data points with the highest influence score.

```bash
python3 -m less.data_selection.matching \
  --train_file_names <str> \
  --ckpts <str> \
  --dims <int> \
  --checkpoint_weights <str> \
  --target_task_names <str> \
  --target_task_files <str> \
  --val_task_load_method <str> \
  --model_path <str>
```
`train_file_names` is a list of training data names that you used to store the gradients.  
`ckpts` is a list of checkpoints, e.g. `'105 211 317 420'`.  
`dims` is the dimension of projection, default is 8192. 
`checkpoint_weights` is a list of average lr of the epoch (check in Wandb), e.g. `'1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06'`.  
`target_task_names` is a list of target task names that you used to store the gradients.   
`target_task_files` can be a full path or a HF repo name, don't forget to specify the `val_task_load_method` accordingly.  
`model_path` is the path to the model in the `out` directory, e.g. `llama2-7b-p0.05-lora-seed3`.

The influence score for each training data point will be saved in the `OUTPUT_PATH` directory. You can use the following script to select the top-k data points with the highest influence score. 

```bash
python3 -m less.data_selection.write_selected_data \
--target_task_names <str> \
--train_file_names <str> \
--train_files <str> \
--output_path <str> \
--percentage <float>
```

### Step 4: Train with your selected data
After selecting the data, you can use the following script to train the model with the selected data. 

```bash 
TARGET_TASK_NAME="tydiqa"
PERCENTAGE=0.05
TRAIN_FILES=../selected_data/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
MODEL_PATH=meta-llama/Llama-2-7b-hf
JOB_NAME=llama2-7b-less-p${PERCENTAGE}-lora

./less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 
```
Note that you can also perform full-parameter finetuning by removing the lora training parameters. 

## Evaluation
Please follow the instructions in the [evaluation](evaluation/README.md) folder to evaluate the performance of the model trained on the selected data.

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Mengzhou (mengzhou@princeton.edu). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@inproceedings{xia2024less,
   title={{LESS}: Selecting Influential Data for Targeted Instruction Tuning},
   author={Xia, Mengzhou and Malladi, Sadhika and Gururangan, Suchin and Arora, Sanjeev and Chen, Danqi},
   booktitle={International Conference on Machine Learning (ICML)},
   year={2024}
}
```




