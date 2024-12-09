import argparse
import os

import torch
from less.data_selection.get_validation_dataset import get_dataset
from tqdm import tqdm

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--train_file_name', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+', help="Checkpoints, e.g. '105 211 317 420'")
argparser.add_argument('--dims', default=8192, type=int, required=False, help='Dimention used for grads')
argparser.add_argument('--checkpoint_weights', type=float, nargs='+', help="Average lr of the epoch (check in wandb)")
argparser.add_argument('--target_task_name', type=str,
                       nargs='+', help="The name of the target task")
argparser.add_argument('--target_task_file', type=str, nargs='+',
                       help='Can be a full path or a HF repo name')
argparser.add_argument('--val_task_load_method', type=str, default=None, help='The method to load the validation data, can be "hf", "local_hf", "local_json"')
argparser.add_argument('--model_path', type=str, required=True, help='Model path, e.g. llama2-7b-p0.05-lora-seed3')

args = argparser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores


# renormalize the checkpoint weights
if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

# calculate the influence score for each validation task
for target_task_name, target_task_file in zip(args.target_task_names, args.target_task_files):
    val_dataset = get_dataset(task=target_task_name, data_dir=target_task_file, 
                              model_path=args.model_path, val_task_load_method=args.val_task_load_method)
    num_val_examples = len(val_dataset)
    for train_file_name in args.train_file_names:
        influence_score = 0
        for i, ckpt in enumerate(args.ckpts):
            # validation_path = args.validation_gradient_path.format(
            # target_task_name, ckpt)
            validation_path = os.path.join("../grads", args.model_path, f"{train_file_name}_ckpt{ckpt}_sgd/dim{args.dims}")
            if os.path.isdir(validation_path):
                validation_path = os.path.join(validation_path, "all_orig.pt")
            validation_info = torch.load(validation_path)

            if not torch.is_tensor(validation_info):
                validation_info = torch.tensor(validation_info)
            validation_info = validation_info.to(device).half()
            # gradient_path = args.gradient_path.format(train_file_name, ckpt)
            gradient_path = os.path.join("../grads", args.model_path, f"{train_file_name}_ckpt{ckpt}_adam/dim{args.dims}")
            if os.path.isdir(gradient_path):
                gradient_path = os.path.join(gradient_path, "all_orig.pt")
            training_info = torch.load(gradient_path)

            if not torch.is_tensor(training_info):
                training_info = torch.tensor(training_info)
            training_info = training_info.to(device).half()

            influence_score += args.checkpoint_weights[i] * \
                calculate_influence_score(
                    training_info=training_info, validation_info=validation_info)
        influence_score = influence_score.cpu().reshape(
            influence_score.shape[0], num_val_examples, -1
        ).mean(-1).max(-1)[0]
        output_dir = os.path.join("../selected_data", target_task_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"{train_file_name}_influence_score.pt")
        torch.save(influence_score, output_file)
        print("Saved influence score to {}".format(output_file))
