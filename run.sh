TRAIN_PATH=datasets/AI.xlsx
TEST_PATH=datasets/AI1000.xlsx
OUTPUT_DIR=output
TRAIN_VALID_SPLIT=0.8
SEED=42
HF_DATASET_NAME=ZH-Reading-Comprehension

rm -rf $OUTPUT_DIR

python create_dataset.py \
    --train_path $TRAIN_PATH \
    --test_path $TEST_PATH \
    --output_dir $OUTPUT_DIR \
    --train_valid_split $TRAIN_VALID_SPLIT \
    --seed $SEED \
    --hf_dataset_name $HF_DATASET_NAME