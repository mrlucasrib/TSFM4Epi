#!/bin/bash
set -e
# List of parquet files
files=(
    "BOTUBR" "DERMBR" "HANSBR" "LERDBR" "PESTBR" "SIFCBR" "TOXGBR"
    "CHAGBR" "DIFTBR" "HANTBR" "LTANBR" "PFANBR" "SIFGBR" "TRACBR"
    "CHIKBR" "ESQUBR" "HEPABR" "PNEUBR" "SRCBR" "VARCBR"
    "COLEBR" "EXANBR" "LEIVBR" "MALABR" "RAIVBR" "TETABR" "ZIKABR"
    "COQUBR" "FMACBR" "LEPTBR" "MENIBR" "SDTABR" "TETNBR"
    "DENGBR" "FTIFBR" "LERBR" "NTRABR" "SIFABR" "TOXCBR"
)

# Define models and their configurations
declare -A models
models=(
    ["lagllama"]="lagllama /lag-llama/lag-llama.ckpt"
    # ["tinytimemixer"]="ttms not-used"
    ["moirai"]="moirai Salesforce/moirai-1.1-R-small"
)
# Create logs directory if it doesn't exist
mkdir -p logs

# Loop through each model
for model_key in "${!models[@]}"; do
    # Split model configuration into name and path
    read -r model_name model_path <<< "${models[$model_key]}"
    docker build -t ${model_name} --build-arg EXPERIMENT_PATH="experiment_${model_name}" . >> "logs/build_logs_${model_name}.txt" 2>&1

    echo "Processing model: $model_name"
    # Check if requirements.txt exists in experiment folder
    # if [ ! -f "experiment_${model_name}/requirements.txt" ]; then
    #     echo "Creating requirements.txt for ${model_name}..."
    #     pip-compile "experiment_${model_name}/requirements.in" base_requirements.in -o experiment_${model_name}/requirements.txt 
    # fi
    # Loop through each file
    for file in "${files[@]}"; do
        echo "Processing $file with $model_name..."
        MLFLOW_TRACKING_URI=/artifacts/mlruns docker run -v ~/data:/data -v ~/artifacts:/artifacts ${model_name} \
            --experiment_name "${file}" \
            --artifacts_path "/artifacts" \
            --prediction_length 24 \
            --context_length 32 \
            --model_name "AutoArima" \
            --data_path "/data/${file}.parquet" \
            --num_samples 100 \
            --model_path "$model_path" >> "logs/logs_${model_name}_${file}.txt" 2>&1
            
        echo "Completed $file with $model_name"
        echo "-------------------"
    done
done