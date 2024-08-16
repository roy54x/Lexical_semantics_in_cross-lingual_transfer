import os
from pathlib import Path
from tokenizers.implementations import ByteLevelBPETokenizer
import itertools


def get_tokenizer(data_dirs, output_name):
    paths = list(itertools.chain.from_iterable([[str(x) for x in Path(data_path).glob(
        '**/*.txt')] for data_path in data_dirs]))
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=30_527, min_frequency=2, special_tokens=[
        '<s>', '<pad>', '</s>', '<unk>', '<mask>'])
    os.mkdir('./'+output_name)
    tokenizer.save_model(output_name)


if __name__ == '__main__':
    get_tokenizer(["data\\cc100_data_clean\\cc100_en", "data\\cc100_data_clean\\cc100_el"], 'tokenizer_en_el_cc100_clean')