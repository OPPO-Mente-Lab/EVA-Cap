#!/bin/bash
MODEL_PATHS="$1"
RESULTS_DIR="$2"

mkdir -p $RESULTS_DIR

ORIGINAL_FILE="eval_scripts/ChronusAV/grouped_data_sub200.json"
MERGED_FILE="$RESULTS_DIR/captioned_results.json"
if [ ! -f "$MERGED_FILE" ]; then
    echo "MERGED_FILE not found. Creating from ORIGINAL_FILE..."
    cp "$ORIGINAL_FILE" "$MERGED_FILE"
fi
CAPTION_FILES_TO_MERGE=()
CAPTION_KEYS=()

# Step 1: caption geneartion
for model_path in "${MODEL_PATHS[@]}"; do
    CLEAN_PATH="${model_path%/}"
    model_name=$(basename "$CLEAN_PATH")

    caption_file="$RESULTS_DIR/${model_name}_caption.jsonl"
    echo "Output caption file will be: $caption_file"

    python eval_scripts/ChronusAV/generate_caption.py \
        --model_path "$model_path" \
        --fout_path "$caption_file" \
        --gpus 8

    if [ -f "$caption_file" ]; then
        CAPTION_FILES_TO_MERGE+=("$caption_file")
        CAPTION_KEYS+=("${model_name}_caption")
    else
        echo "Error: Caption file $caption_file not generated for model $model_path."
        exit 1
    fi
done

# Step 2: merge generated caption files
echo "Merging all generated caption files..."
python eval_scripts/ChronusAV/merge_captions.py \
    --original_file "$MERGED_FILE" \
    --caption_files "${CAPTION_FILES_TO_MERGE[@]}" \
    --merged_file "$MERGED_FILE"

# Step 3: evaluation
python eval_scripts/ChronusAV/evaluation_gemini.py \
    --merged_file "$MERGED_FILE" \
    --caption_keys "${CAPTION_KEYS[@]}"

# Step 4: analysis and save evaluation results
for caption_key in "${CAPTION_KEYS[@]}"; do
    echo "Running analysis for caption key: $caption_key"

    result_file="$RESULTS_DIR/${caption_key}_result_gemini.jsonl"
    if [ -f "$result_file" ]; then
        python eval_scripts/ChronusAV/cal_caption.py --result_file_path "$result_file"
    else
        echo "Warning: Result file '$result_file' not found for analysis."
    fi
done

# vdc
python eval_scripts/ChronusAV/vdc_seed.py --merged_file "$result_file"
