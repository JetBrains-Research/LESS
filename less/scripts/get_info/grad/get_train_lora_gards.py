import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_name", type=str, required=True, help="Name of the training data (will be used to store gradients)")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model in the `out` directory")
    parser.add_argument("--ckpts", type=str, required=True, help="List of checkpoints to compute gradients for")
    parser.add_argument("--dims", default=8192, type=int, required=False, help="Dimension of projection")
    args = parser.parse_args()

    # Split checkpoint string into list of ints
    ckpts = [int(x) for x in args.ckpts.split()]

    # Process each checkpoint
    for ckpt in ckpts:
        # Construct model path with checkpoint
        model = os.path.join("../out", args.model_path, f"/checkpoint-{ckpt}")
        
        # Construct output path with checkpoint
        output_path = os.path.join("../grads", args.model_path, f"{args.train_data_name}-ckpt{ckpt}-adam")

        # Store paths for use in command
        args.model = model
        args.output_path = output_path

        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Build command
        cmd = [
            "python3", "-m", "less.data_selection.get_info",
            "--train_file", args.train_file,
            "--info_type", "grads",
            "--model_path", args.model,
            "--output_path", args.output_path,
            "--gradient_projection_dimension", args.dims,
            "--gradient_type", "adam"
        ]

        subprocess.run(cmd)

if __name__ == "__main__":
    main()

