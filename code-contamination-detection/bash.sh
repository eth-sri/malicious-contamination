#!/bin/bash

# Define arrays for datasets and models
datasets=("truthful_qa" "cais/mmlu" "gsm8k" "ai2_arc")
models=("microsoft/phi-2" "gpt2-xl" "mistralai/Mistral-7B-v0.1")

# Create an associative array for mapping
declare -A dataset_map
dataset_map["gsm8k"]="gsm8k"
dataset_map["ai2_arc"]="arc"
dataset_map["cais/mmlu"]="mmlu"
dataset_map["truthful_qa"]="truthfulqa"

# Loop over models
for model in "${models[@]}"; do
    # Loop over datasets
    for dataset in "${datasets[@]}"; do
        # Define target models array dynamically based on the dataset
        target_models=("test/${dataset_map[$dataset]}/0" "test/${dataset_map[$dataset]}/1" "seed/0" "test/${dataset_map[$dataset]}/epochs_1/0" "test/${dataset_map[$dataset]}/epochs_1/1")
        # source_file=output/${model}/testv2/0/${dataset_map[$dataset]}/generated_0.csv;

        # Loop over target models
        for target_model in "${target_models[@]}"; do
            if [ "$target_model" == "seed/0" ]; then
                source_file=output/${model}/test/${dataset_map[$dataset]}/0/generated_0.csv;
            else
                source_file=output/${model}/${target_model}/generated_0.csv;
            fi
            # Define the output directory
            output_dir="code-contamination-output/${model//\//-}_${dataset}_${target_model//\//-}"

            # Check if the output directory already exists
            if [ -f "$output_dir/all_output.jsonl" ]; then
                echo "Output directory ${output_dir} already exists, skipping..."
                continue
            fi

            # Create the output directory
            mkdir -p "$output_dir"
            # Run the command
            python code-contamination-detection/src/run.py --target_model "output/${model}/${target_model}" \
                --ref_model "${model}" \
                --data "$dataset" \
                --output_dir "$output_dir" \
                --ratio_gen 0.4 \
                --was_trained_source "${source_file}" > "${output_dir}/log.txt"
        done
    done
done
