from abc import ABC, abstractmethod
from transformers import AutoTokenizer
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
    text: str


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_tsv(data: list[Data], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('id\tinstruction\toutput\ttext\n')
        for d in data:
            f.write(f'{d.id}\t{d.instruction}\t{d.output}\t{d.text}\n')


class TextProcessor(ABC):
    def preprocess_text(self, text: str) -> str:
        text = str(text)
        text = text.replace('\n', '')
        text = text.replace('\t', '')
        text = text.replace('x0000', '')
        text = text.replace('_____________________x0006_____䋊__䎴ꀀ䒤__䍺____Ṱ෰_____x0005________________________________x0003_____䋊耀䎃__䏞__䈴____Ṱ෰_____x0001_______衐ෟ_____________________x001B_____________䒴耀䐡____Ṱ෰_____x0005_______衐ෟ_____________________x0003_____䆀耀䐚__䍓__䇠____Ṱ෰_____x0001_______________________________________________________________징', '')
        return text

    def preprocess_output_text(self, text: str) -> str:
        text = self.preprocess_text(text)
        text = text.replace('選項', '')
        return text

    @abstractmethod
    def use_tokenizer_chat_template(self, instruction: str, output: str) -> str:
        pass


class LlamaTextProcessor(TextProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def use_tokenizer_chat_template(self, instruction: str, output: str) -> str:
        # test
        if output == "-1":
            chat = [
                {"role": "user", "content": instruction}
            ]
            text = self.tokenizer.apply_chat_template(chat, tokenize=False)
            text += "<|start_header_id|>assistant<|end_header_id|>"
        # train or valid
        else:
            chat = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
            text = self.tokenizer.apply_chat_template(chat, tokenize=False)
            text += self.tokenizer.eos_token
        return self.preprocess_text(text)


class BreezeTextProcessor(TextProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def use_tokenizer_chat_template(self, instruction: str, output: str) -> str:
        # test
        if output == "-1":
            chat = [
                {"role": "user", "content": instruction}
            ]
            text = self.tokenizer.apply_chat_template(chat, tokenize=False)
        # train or valid
        else:
            chat = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
            text = self.tokenizer.apply_chat_template(chat, tokenize=False)
            text += self.tokenizer.eos_token
        return text


def get_text_processor(tokenizer, tokenizer_name) -> TextProcessor:
    if tokenizer_name == "meta-llama/Meta-Llama-3-8B-Instruct":
        return LlamaTextProcessor(tokenizer)
    elif tokenizer_name == "MediaTek-Research/Breeze-7B-Instruct-v1_0":
        return BreezeTextProcessor(tokenizer)
    else:
        raise ValueError(f"Unknown tokenizer_name: {tokenizer_name}")


def main(args):
    random.seed(args.seed)

    create_dir(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    text_processor = get_text_processor(tokenizer, args.tokenizer_name)

    # Train dataset
    df = pd.read_excel(args.train_path)
    original_train_data: list[Data] = []
    for i in range(0, len(df)):
        document = text_processor.preprocess_text(df['文章'][i])
        question = text_processor.preprocess_text(df['問題'][i])
        item1 = text_processor.preprocess_text(df['選項1'][i])
        item2 = text_processor.preprocess_text(df['選項2'][i])
        item3 = text_processor.preprocess_text(df['選項3'][i])
        item4 = text_processor.preprocess_text(df['選項4'][i])
        answer = text_processor.preprocess_output_text(df['正確答案'][i])
        instruction = f'{INSTRUCTION_START};說明:{document};問題:{question}1:{item1},2:{item2},3:{item3},4:{item4}'
        original_train_data.append(Data(
            id=str(i),
            instruction=instruction,
            output=answer,
            text=text_processor.use_tokenizer_chat_template(
                instruction, answer)
        ))

    random.shuffle(original_train_data)
    train_data_length = int(len(original_train_data)*args.train_valid_split)

    write_tsv(original_train_data[:train_data_length],
              f'{args.output_dir}/train.tsv')
    write_tsv(original_train_data[train_data_length:],
              f'{args.output_dir}/validation.tsv')

    # Test dataset
    df = pd.read_excel(args.test_path)
    test_data: list[Data] = []
    for i in range(0, len(df)):
        id = text_processor.preprocess_text(df['題號'][i])
        document = text_processor.preprocess_text(df['文章'][i])
        question = text_processor.preprocess_text(df['問題'][i])
        item1 = text_processor.preprocess_text(df['選項1'][i])
        item2 = text_processor.preprocess_text(df['選項2'][i])
        item3 = text_processor.preprocess_text(df['選項3'][i])
        item4 = text_processor.preprocess_text(df['選項4'][i])
        instruction = f'{INSTRUCTION_START};說明:{document};問題:{question}1:{item1},2:{item2},3:{item3},4:{item4}'
        test_data.append(Data(
            id=id,
            instruction=instruction,
            output="-1",
            text=text_processor.use_tokenizer_chat_template(
                instruction, "-1"))
        )

    write_tsv(test_data,
              f'{args.output_dir}/test.tsv')

    # Push to Hugging Face Dataset Hub
    dataset: Dataset = load_dataset(args.output_dir)
    dataset.push_to_hub(args.hf_dataset_name)
    print(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, required=True)
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
