import random
import pandas as pd
import json
import argparse
import os

from dataclasses import dataclass
from datasets import load_dataset, Dataset

INSTRUCTION_START = '請根據以下輸入回答選擇題，並以數字回答:'


@dataclass
class Data:
    id: str
    instruction: str
    output: str


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def preprocess_text(text) -> str:
    text = str(text)
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    return text


def write_tsv(data: list[Data], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('id\tinstruction\toutput\n')
        for d in data:
            f.write(f'{d.id}\t{d.instruction}\t{d.output}\n')


def main(args):
    random.seed(args.seed)

    train_path = args.train_path
    test_path = args.test_path
    output_dir = args.output_dir
    train_valid_split = args.train_valid_split
    create_dir(output_dir)

    # Train dataset
    df = pd.read_excel(train_path)
    original_train_data: list[Data] = []
    for i in range(0, len(df)):
        document = preprocess_text(df['文章'][i])
        question = preprocess_text(df['問題'][i])
        item1 = preprocess_text(df['選項1'][i])
        item2 = preprocess_text(df['選項2'][i])
        item3 = preprocess_text(df['選項3'][i])
        item4 = preprocess_text(df['選項4'][i])
        answer = preprocess_text(df['正確答案'][i])
        original_train_data.append(Data(
            id=str(i),
            instruction=f'{INSTRUCTION_START};說明:{document};問題:{question}1:{item1},2:{item2},3:{item3},4:{item4}',
            output=answer
        ))

    random.shuffle(original_train_data)
    train_data_length = int(len(original_train_data)*train_valid_split)

    write_tsv(original_train_data[:train_data_length],
              f'{output_dir}/train.tsv')
    write_tsv(original_train_data[train_data_length:],
              f'{output_dir}/validation.tsv')

    # Test dataset
    df = pd.read_excel(test_path)
    test_data: list[Data] = []
    for i in range(0, len(df)):
        document = preprocess_text(df['文章'][i])
        question = preprocess_text(df['問題'][i])
        item1 = preprocess_text(df['選項1'][i])
        item2 = preprocess_text(df['選項2'][i])
        item3 = preprocess_text(df['選項3'][i])
        item4 = preprocess_text(df['選項4'][i])
        test_data.append(Data(
            id=str(df['題號'][i]),
            instruction=f'{INSTRUCTION_START};說明:{document};問題:{question}1:{item1},2:{item2},3:{item3},4:{item4}',
            output="-1"
        ))

    write_tsv(test_data,
              f'{output_dir}/test.tsv')

    # Push to Hugging Face Dataset Hub
    dataset: Dataset = load_dataset(output_dir)
    dataset.push_to_hub(args.hf_dataset_name)
    print(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_path',
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        '--test_path',
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        '--output_dir',
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        '--train_valid_split',
        default=None,
        type=float,
        required=True)
    parser.add_argument(
        '--seed',
        type=int,
        required=True
    )
    parser.add_argument(
        '--hf_dataset_name',
        type=str,
        required=True
    )
    main(parser.parse_args())
