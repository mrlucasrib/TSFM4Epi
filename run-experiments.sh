#!/bin/bash
set -o pipefail
set -e
echo "Script PID: $$"
export GIT_PYTHON_REFRESH=quiet
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
    ["lagllama"]="lagllama /models/lag-llama.ckpt"
    ["auto_arima"]="baseline not-used"
    ["auto_theta"]="baseline not-used"
    ["auto_regressive"]="baseline not-used"
    ["auto_ces"]="baseline not-used"
    ["seasonal_naive"]="baseline not-used"
    ["naive"]="baseline not-used"
    ["auto_ets"]="baseline not-used"
    ["croston_classic"]="baseline not-used"
    ["croston_optimized"]="baseline not-used"
    ["croston_sba"]="baseline not-used"
    ["tinytimemixer-r1"]="ttms granite-timeseries-ttm-r1"
    ["tinytimemixer-r2"]="ttms granite-timeseries-ttm-r2"
    ["timesfm-1.0"]="timesfm timesfm-1.0-200m"
    ["timesfm-2.0"]="timesfm timesfm-2.0-500m-jax"
    ["moirai-small"]="moirai moirai-1.1-R-small"
    ["moirai-base"]="moirai moirai-1.1-R-base"
    ["moirai-large"]="moirai moirai-1.1-R-large"
    ["moirai-moe-small"]="moirai moirai-moe-1.0-R-small"
    ["moirai-moe-base"]="moirai moirai-moe-1.0-R-base"
    ["chronos-bolt-tiny"]="chronos chronos-bolt-tiny"
    ["chronos-bolt-mini"]="chronos chronos-bolt-mini"
    ["chronos-bolt-small"]="chronos chronos-bolt-small"
    ["chronos-bolt-base"]="chronos chronos-bolt-base"
    ["chronos-bolt-large"]="chronos chronos-bolt-large"
    ["chronos-t5-tiny"]="chronos chronos-bolt-tiny"
    ["chronos-t5-mini"]="chronos chronos-bolt-mini"
    ["chronos-t5-small"]="chronos chronos-bolt-small"
    ["chronos-t5-base"]="chronos chronos-bolt-base"
    ["chronos-t5-large"]="chronos chronos-t5-large"
)
# Create logs directory if it doesn't exist
mkdir -p logs

# Get model keys, store in variable and sort them
model_keys=($(for key in "${!models[@]}"; do echo "$key ${models[$key]}" | awk '{print $1, $2}' ; done | sort -k2,2 | awk '{print $1}'))

# Loop through each model
previous_model_name=""
for model_key in "${model_keys[@]}"; do
    # Split model configuration into name and path
    read -r model_name model_path <<< "${models[$model_key]}"
    
    # If model name changes, clean up Docker containers and images
    if [[ "$model_name" != "$previous_model_name" && -n "$previous_model_name" ]]; then
        docker rm $(docker ps -a -q --filter ancestor=${previous_model_name}) && docker rmi ${previous_model_name}
        docker builder prune -f
    fi
    
    docker build -t ${model_name} --build-arg EXPERIMENT_PATH="experiment_${model_name}" . 2>&1 | tee "logs/build_logs_${model_name}.txt"
    echo "Processing model: $model_key"
    
    # Loop through each file
    for file in "${files[@]}"; do
        echo "Processing $file with $model_key..."
        MLFLOW_TRACKING_URI=/artifacts4/mlruns GIT_PYTHON_REFRESH=quiet docker run --gpus all -v ~/data:/data -v ~/artifacts:/artifacts ${model_name} \
            --experiment_name "${file}" \
            --artifacts_path "/artifacts" \
            --prediction_length 24 \
            --context_length 32 \
            --model_name "$model_key" \
            --data_path "/data/${file}.parquet" \
            --num_samples 100 \
            --model_path "$model_path" 2>&1 | tee "logs/logs_${model_key}_${file}.txt" || true
        echo "Completed $file with $model_key"
        echo "-------------------"
    done
    
    previous_model_name="$model_name"
done

# Final cleanup
docker rm $(docker ps -a -q --filter ancestor=${previous_model_name}) && docker rmi ${previous_model_name}
docker builder prune -f