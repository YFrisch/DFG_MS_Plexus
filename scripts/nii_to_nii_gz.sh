#!/bin/bash

# Helper script to convert .nii files to .nii.gz

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_directory output_directory"
    exit 1
fi

input_dir="$1"
output_dir="$2"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through all .nii files in the input directory
for file in "$input_dir"/*.nii; do
    # Check if the file exists
    if [ -e "$file" ]; then
        # Get the base filename without extension
        base_name=$(basename "$file" .nii)
        # Compress the .nii file to .nii.gz in the output directory
        gzip -c "$file" > "$output_dir/$base_name.nii.gz"
        echo "Converted $file to $output_dir/$base_name.nii.gz"
    else
        echo "No .nii files found in $input_dir."
        break
    fi
done