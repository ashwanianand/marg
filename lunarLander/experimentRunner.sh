#!/bin/bash



# Parse optional --tag argument
TAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Run record_videos script, optionally with --tag
echo "Starting video recording..."
if [ -n "$TAG" ]; then
    python experiments/record_videos.py --tag "$TAG"
else
    python experiments/record_videos.py
fi

# Check if record_videos completed successfully
if [ $? -eq 0 ]; then
    echo "Video recording completed successfully."
    
    # Run report_generator script, optionally with --tag
    echo "Generating reports..."
    if [ -n "$TAG" ]; then
        python experiments/report_generator.py --tag "$TAG"
        python experiments/graph_generator.py --tag "$TAG"
    else
        python experiments/report_generator.py
        python experiments/graph_generator.py
    fi
    
    # Check if report_generator completed successfully
    if [ $? -eq 0 ]; then
        echo "Report generation completed successfully."
        echo "Experiment runner finished successfully!"
    else
        echo "Error: Report generation failed."
        exit 1
    fi
else
    echo "Error: Video recording failed."
    exit 1
fi


