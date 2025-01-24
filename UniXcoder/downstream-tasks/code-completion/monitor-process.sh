#!/bin/bash

# Arguments
PID=$1
LOG_FILE=$2
START_TIME_SECONDS=$3

# Monitor the process
while kill -0 $PID 2>/dev/null; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Training is still running..." >> "$LOG_FILE"
    sleep 60  # Check every 60 seconds
done

# Record the completion time
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
END_TIME_SECONDS=$(date +%s)
DURATION=$((END_TIME_SECONDS - START_TIME_SECONDS))

# Convert duration to HH:MM:SS format
DURATION_FORMAT=$(printf '%02d:%02d:%02d\n' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60)))

echo "Training completed at: $END_TIME" >> "$LOG_FILE"
echo "Total duration: $DURATION_FORMAT (HH:MM:SS)" >> "$LOG_FILE"
