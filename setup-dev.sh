#!/bin/bash

# Loop through all directories starting with "experiment-"
for dir in experiment-*; do
    if [ -d "$dir" ]; then
        new_dir=$(echo "$dir" | sed 's/-/_/g')
        # mv "$dir" "$new_dir"
        # cd "$new_dir"
        # echo "$dir"
        # Create virtual environment
        # if [ ! -d ".venv" ]; then
        #     virtualenv --prompt . --symlink-app-data .venv
        
        
        #     # Activate the virtual environment
        # source .venv/bin/activate

        #     if [ -f "requirements.txt" ]; then
        pip install -r requirements.in
        #     fi
        deactivate
        # fi
        # # Create requirements.in file
        # touch requirements.in
        # echo "datasets" >> requirements.in
        # Install dependencies if requirements.txt exists
        
        # Go back to the parent directory
        cd ..
    fi
done


python -m experiment_morai --experiment_name "exp1" \
              --artifacts_path "artifacts" \
              --prediction_length 24 \
              --context_length 32 \
              --ckpt_path "/path/to/checkpoint" \
              --model_name "moirai" \
              --data_path "data/sinan/newdata/ANIMBR_municipality_M_train.parquet" \
              --num_samples 100 \
              --model_size "small"


python -m experiment_chronos --experiment_name "exp3" \
              --artifacts_path "artifacts" \
              --prediction_length 24 \
              --context_length 32 \
              --ckpt_path "/path/to/checkpoint" \
              --model_name "chronos-base" \
              --data_path "data/sinan/newdata/ANIMBR_municipality_M_train.parquet" \
              --num_samples 100 \
              --model_path "amazon/chronos-bolt-base"

Salesforce/moirai-1.1-R-{args.model_path}

python -m experiment_chronos --experiment_name "exp3" \
              --artifacts_path "artifacts" \
              --prediction_length 24 \
              --context_length 32 \
              --ckpt_path "/path/to/checkpoint" \
              --model_name "chronos-t5" \
              --data_path "data/sinan/newdata/ANIMBR_municipality_M_train.parquet" \
              --num_samples 100 \
              --model_path "amazon/chronos-t5-small"


python -m experiment_baseline --experiment_name "exp4" \
              --artifacts_path "artifacts" \
              --prediction_length 24 \
              --context_length 32 \
              --ckpt_path "/path/to/checkpoint" \
              --model_name "auto_arima" \
              --data_path "data/sinan/newdata/ANIMBR_municipality_M_train.parquet" \
              --num_samples 100 \
              --model_path "amazon/chronos-t5-small"

docker run -v /media/work/lucasribeiro/GitProjects/Ribeiro2025/data:/data -v ./artifacts:/workspaces/artifacts moirai --experiment_name "exp1" \
              --artifacts_path "artifacts" \
              --prediction_length 24 \
              --context_length 32 \
              --model_name "moirai" \
              --data_path "/data/processed_gluonts/VARCBR.parquet" \
              --num_samples 100 \
              --model_path "Salesforce/moirai-1.1-R-small"


docker run -v ~/data:/data -v ~/artifacts:/artifacts tinytimemixer \
            --experiment_name "exp1" \
            --artifacts_path "/artifacts" \
            --prediction_length 24 \
            --context_length 32 \
            --model_name "tinytimemixer" \
            --data_path "/data/VARCBR.parquet" \
            --num_samples 100 \
            --model_path "not-used"