#!/bin/bash

# Parse flags
steps=""
dataset=""
while getopts "s:d:h" opt; do
    case $opt in
        s) steps="$OPTARG" ;;
        d) dataset="$OPTARG" ;;
        h)
            echo "Usage: $0 [-s steps] [-d dataset]"
            exit 0
            ;;
        *) 
            echo "Usage: $0 [-s steps] [-d dataset]"
            exit 1
            ;;
    esac
done

# Set defaults if not provided
[ -z "$steps" ] && steps=100000
[ -z "$dataset" ] && dataset="."

datestamp=$(date +"%Y%m%d%H%M%S")
out_dir="results"
log_file="logs_$datestamp.log"

# Create a dataset called results_steps if it does not exist already
if [ ! -d "$out_dir" ]; then
    mkdir "$out_dir"
    mkdir "$out_dir/outputs"
    touch "$out_dir/done.txt"
    touch "$out_dir/skipped_files.txt"
    touch "$out_dir/$log_file"
    echo -e "\\e[34mDirectory "$out_dir" created.\\e[0m"
fi


export steps out_dir log_file # Export these variables for use in parallel

process_file() {
    file=$1
    # log the file being processed with the start time
    echo -e "\\nProcessing $(basename "$file") at $(date +"%H:%M")" >> "$out_dir/$log_file"



    # Check if the file is in done.txt
    if grep -Fxq "$(basename "$file")" "$out_dir/done.txt"; then
        echo -e "\\t\\e[32mAlready processed!\\e[0m"
        echo -e "Already processed $(basename "$file")" >> "$out_dir/$log_file"
        return
    fi

    output_file_temp="$out_dir/outputs/$(basename "$file").json"
    std_out_file="$out_dir/outputs/$(basename "$file").out"
    
    # Construct the java command
    CMD="python3 experiments/evaluateShield.py $steps \"$file\" \"$output_file_temp\" >> \"$std_out_file\""

    # Execute the command
    eval $CMD

    echo "$(basename "$file")" >> "$out_dir/done.txt"
    # log the file being processed with the end time
    echo -e "Finished processing $(basename "$file") at $(date +"%H:%M")" >> "$out_dir/$log_file"
}

export -f process_file  # Export function for parallel

# Find all grid files in .json and process them in parallel with a progress bar
total_files=$(find "$dataset" -type f -name "*.json" | wc -l)
if [ "$total_files" -eq 0 ]; then
    echo "No .json files found in the dataset!"
    exit 1
fi

#find "$dataset" -type f -name "*.json" | parallel --no-notice --bar -j$(nproc) process_file
find "$dataset" -type f -name "*.json" ! -path "$dataset/temp/*" | parallel --no-notice --bar -j$(nproc) process_file


# # After the experiment is done, create the plots
echo -e "\\n\\e[34mCreating plots...\\e[0m"
python3 experiments/generate_plot.py