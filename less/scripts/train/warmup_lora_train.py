import subprocess
from pathlib import Path
import argparse
import random  

def run_training(
    train_file: str,
    model_name_or_path: str, 
    percentage: float=0.05,
    data_seed: int=3,
):
    job_name = f"{model_name_or_path.split('/')[-1]}-p{percentage}-lora-seed{data_seed}"
    output_dir = Path("../out") / job_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base training args from base_training_args.sh
    base_training_args = [
    "--do_train", "True",
    "--max_seq_length", "2048",
    "--use_fast_tokenizer", "True",
    "--lr_scheduler_type", "linear",
    "--warmup_ratio", "0.03",
    "--weight_decay", "0.0",
    "--evaluation_strategy", "no",
    "--logging_steps", "1",
    "--save_strategy", "epoch",
    "--num_train_epochs", "4",
    "--bf16", "True",
    "--tf32", "False",
    "--fp16", "False",
    "--overwrite_output_dir", "True",
    "--report_to", "wandb",
    "--optim", "adamw_torch",
    "--lora", "True",
    "--lora_r", "128",
    "--lora_alpha", "512",
    "--lora_dropout", "0.1",
    "--lora_target_modules", "q_proj", "k_proj", "v_proj", "o_proj",
    "--learning_rate", "2e-05",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "32"
]

    # Add FSDP config for large models
    if model_name_or_path == "meta-llama/Llama-2-13b-hf":
        base_training_args.extend([
            "--fsdp", "full_shard auto_wrap",
            "--fsdp_config", "llama2_13b_finetune"
        ])
    elif model_name_or_path == "mistralai/Mistral-7B-v0.1":
        base_training_args.extend([
            "--fsdp", "full_shard auto_wrap", 
            "--fsdp_config", "mistral_7b_finetune"
        ])

    # Combine all training args
    training_args = [
        "--model_name_or_path", model_name_or_path,
        "--output_dir", str(output_dir),
        "--percentage", str(percentage),
        "--data_seed", str(data_seed),
        "--train_files"
    ] + [train_file] + base_training_args

    # Set up logging
    log_file = output_dir / "train.log"
    with open(log_file, "w") as f:
        print(training_args)
        # Run training command
        process = subprocess.Popen(
            ["torchrun", "--nproc_per_node", "1", "--nnodes", "1", "--rdzv-id", str(random.randint(0,65535)), "--rdzv_backend", "c10d", "-m", "less.train.train"] + training_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output to console and log file
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            
        process.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--percentage", type=float, default=0.05)
    parser.add_argument("--data_seed", type=int, default=3)
    args = parser.parse_args()
    run_training(args.train_file, args.model_name_or_path, args.percentage, args.data_seed)