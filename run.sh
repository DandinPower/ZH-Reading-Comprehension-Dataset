TRAIN_PATH=datasets/AI.xlsx
TEST_PATH=datasets/AI1000.xlsx
OUTPUT_DIR=output
TRAIN_VALID_SPLIT=0.8
SEED=42
HF_DATASET_NAME=ZH-Reading-Comprehension-TAIDE-Chat
# TOKENIZER_NAME=MediaTek-Research/Breeze-7B-Instruct-v1_0
# TOKENIZER_NAME=meta-llama/Meta-Llama-3-8B-Instruct
TOKENIZER_NAME=taide/TAIDE-LX-7B-Chat
# TOKENIZER_NAME=google/gemma-1.1-7b-it

rm -rf $OUTPUT_DIR

python create_dataset.py \
    --tokenizer_name $TOKENIZER_NAME \
    --train_path $TRAIN_PATH \
    --test_path $TEST_PATH \
    --output_dir $OUTPUT_DIR \
    --train_valid_split $TRAIN_VALID_SPLIT \
    --seed $SEED \
    --hf_dataset_name $HF_DATASET_NAME