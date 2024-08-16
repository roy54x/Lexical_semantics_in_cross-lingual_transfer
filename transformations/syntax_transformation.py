import re
from trankit import Pipeline

from reordering_package.ud_reorder_algo import UdReorderingAlgo
from utils.utils import transform_all_gts, post_process_sentence

pipeline = Pipeline('english')
reorderer = UdReorderingAlgo(UdReorderingAlgo.ReorderAlgo.HUJI)


def transform_sentences_syntax(sentences, language):
    transformed_sentences = []
    for sentence_idx, sentence in enumerate(sentences):
        print(sentence_idx, sentence)
        tokenized_sentence = preprocess_sentence(sentence)
        print(sentence_idx, tokenized_sentence)
        try:
            reorder_mapping = reorderer.get_reorder_mapping(sent=tokenized_sentence,
                                                        reorder_by_lang=language)
            reordered_sentence = reorderer.reorder_sentence(tokenized_sentence,
                                                            reorder_mapping)
            reordered_sentence = post_process_sentence(post_process_sentence(reordered_sentence))
        except Exception:
            reordered_sentence = sentence
        print(sentence_idx, reordered_sentence)
        print()
        transformed_sentences.append(reordered_sentence)
    return transformed_sentences


def preprocess_sentence(sentence):
    sentence_split = sentence.split(" ")
    for i, token in enumerate(sentence_split):
        if "-" in token:
            if token == "-":
                pass
            elif re.findall("\d", token):
                sentence_split[i] = token.replace("-", "")
            else:
                sentence_split[i] = token.replace("-", " ")
    sentence = " ".join(sentence_split)
    tokenized_sentence_split = [token["text"] for token in pipeline.tokenize(sentence, is_sent=True)["tokens"]]
    tokenized_sentence = " ".join(tokenized_sentence_split)
    return tokenized_sentence


if __name__ == '__main__':
    transform_all_gts(r"C:\Users\RoyIlani\pythonProject\data\TED_data\TED_en-zh_cn_gt\train_set",
                      "", transform_sentences_syntax, "hi_syntax_transformation", language="hindi")

    transform_all_gts(r"C:\Users\RoyIlani\pythonProject\data\TED_data\TED_en-zh_cn_gt\test_set",
                      "", transform_sentences_syntax, "hi_syntax_transformation", language="hindi")




