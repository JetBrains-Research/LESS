import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g. tydiqa, mmlu), will be used to store the gradients")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory, can also be a full path or a HF repo name") 
    parser.add_argument("--val_task_load_method", type=str, required=True, help="The method to load the validation data, can be 'hf', 'local_hf', 'local_json'")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model in the `out` directory, e.g. 'llama2-7b-p0.05-lora-seed3'")
    parser.add_argument("--ckpts", type=str, required=True, help="List of checkpoints to compute gradients for, e.g. '105 211 317 420'")
    parser.add_argument("--dims", default=8192, type=int, required=False, help="Dimension of projection")
    args = parser.parse_args()

     # Split checkpoint string into list of ints
    ckpts = [int(x) for x in args.ckpts.split()]

    # Process each checkpoint
    for ckpt in ckpts:
        # Create output directory if it doesn't exist
        model = os.path.join("../out", args.model_path, f"checkpoint-{ckpt}")
        
        # Construct output path with checkpoint
        output_path = os.path.join("../grads", args.model_path, f"{args.task}-ckpt{ckpt}-sgd")

        # Build command
        cmd = [
            "python3", "-m", "less.data_selection.get_info",
            "--task", args.task,
            "--info_type", "grads", 
            "--model_path", model,
            "--output_path", output_path,
            "--gradient_projection_dimension", str(args.dims),
            "--gradient_type", "sgd",
            "--data_dir", args.data_dir,
            "--val_task_load_method", args.val_task_load_method
        ]

        subprocess.run(cmd)

if __name__ == "__main__":
    main()
