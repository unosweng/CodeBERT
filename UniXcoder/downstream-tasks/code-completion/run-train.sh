#!/bin/bash

# Define the log file
LOG_FILE="run-train.log"

# Record the start time
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "Training started at: $START_TIME" >> "$LOG_FILE"

# Start time in seconds for duration calculation
START_TIME_SECONDS=$(date +%s)

# Run the training script with nohup and log output
nohup python run.py \
    --do_train \
    --do_eval \
    --lang java \
    --model_name_or_path microsoft/unixcoder-base \
    --train_filename dataset/javaCorpus/train.txt \
    --dev_filename dataset/javaCorpus/dev.json \
    --output_dir saved_models/javaCorpus \
    --max_source_length 936 \
    --max_target_length 64 \
    --beam_size 5 \
    --train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 10 >> "$LOG_FILE" 2>&1 &

# Get the PID of the nohup process
PID=$!
echo "Training process running in the background with PID: $PID" >> "$LOG_FILE"

# Start the monitor script in the background
nohup ./monitor-process.sh $PID $LOG_FILE $START_TIME_SECONDS >> "$LOG_FILE" 2>&1 &
