#!/bin/bash

FILE=$1

start_time=$(date +%s)

while true
do
    echo "Starting experiment $FILE at $(date)"
    
    python3 "$FILE"
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Program finished successfully."
        break
    fi

    echo "Process crashed with exit code $exit_code at $(date)"
    echo "Restarting in 10 seconds..."
    
    sleep 10
done


end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$((elapsed % 60))

printf "Total execution time: %02d:%02d:%02d\n" $hours $minutes $seconds