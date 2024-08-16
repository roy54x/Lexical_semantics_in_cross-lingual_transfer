import lzma
import os


def save_sentences_to_dir(dataset_path, output_name, limit=-1):
    file = lzma.open(dataset_path, mode="rt",encoding="utf-8")
    os.mkdir(output_name)
    text_data = []
    file_count = 0
    for sentence in file:
        text_data.append(sentence)
        if len(text_data) == 10_000:
            with open(f'{output_name}/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1
            if file_count == limit:
                break


def parse_data(doc):
    sentences = doc.strip().split('\n')
    sentences = [sentence.lower() for sentence in sentences]
    return sentences


if __name__ == '__main__':
    save_sentences_to_dir("data\\cc100_data\\el.txt.xz", "data\\cc100_data\\cc100_el", 20000)
