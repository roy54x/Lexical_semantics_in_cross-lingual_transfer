import re

from trankit import Pipeline
import jieba


def tokenize_sentences(sentences, language, pipeline=None):
    if not pipeline:
        pipeline = Pipeline(language, gpu=True)
    else:
        pipeline.set_active(language)

    tokenized_sentences = []
    for sentence in sentences:
        try:
            if not sentence:
                raise Exception
            if language == "chinese":
                tokenized_sentence = [token_tuple[0] for token_tuple in jieba.tokenize(sentence)]
                tokenized_sentence = list(filter(lambda token: token != "" and not re.match("\s", token), tokenized_sentence))
                tokenized_sentence = [(token, token) for token in tokenized_sentence]
            else:
                tokenized_sentence = []
                lemmatized_sentence = pipeline.lemmatize(sentence, is_sent=True)["tokens"]
                for token in lemmatized_sentence:
                    if isinstance(token["id"], int):
                        tokenized_sentence.append((token["text"], token["lemma"]))
                    else:
                        tokenized_sentence.extend([(sub_token["text"], sub_token["lemma"])
                                           for sub_token in token['expanded']])
        except Exception:
            tokenized_sentences.append(None)
            print("Error tokenizing sentence")
            print()
            continue
        tokenized_sentences.append(tokenized_sentence)
    return tokenized_sentences


def light_tokenize_sentences(sentences, language):
    tokenized_sentences = []
    for sentence in sentences:
        if not sentence:
            tokenized_sentences.append(None)
            print("Error tokenizing sentence")
            print()
            continue
        if language == "chinese":
            tokenized_sentence = [token_tuple[0] for token_tuple in jieba.tokenize(sentence)]
        else:
            tokenized_sentence = sentence.lower()
            tokenized_sentence = tokenized_sentence.replace(".", " .")
            tokenized_sentence = tokenized_sentence.replace(",", " ,")
            tokenized_sentence = tokenized_sentence.replace("?", " ?")
            tokenized_sentence = tokenized_sentence.replace(";", " ;")
            tokenized_sentence = tokenized_sentence.split(" ")
        tokenized_sentence = list(filter(lambda token: token != "" and not re.match("\s", token), tokenized_sentence))
        tokenized_sentences.append(tokenized_sentence)
    return tokenized_sentences