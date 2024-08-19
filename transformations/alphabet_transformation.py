from utils.utils import transform_all_gts


def transform_sentences_alphabet(sentences, language="greek"):
    transformed_sentences = []
    for sentence_idx, sentence in enumerate(sentences):
        print(sentence_idx, sentence)
        try:
            transformed_sentence = swap_letters(sentence, language)
        except Exception:
            transformed_sentence = sentence
        print(sentence_idx, transformed_sentence)
        print()
        transformed_sentences.append(transformed_sentence)
    return transformed_sentences


def swap_letters(sentence, language="greek"):
    if language == "greek":
        letters_list = list(sentence.lower())
        for i in range(len(letters_list)):
            if letters_list[i] == "a":
                letters_list[i] = "α"
            elif letters_list[i] == "b":
                letters_list[i] = "β"
            elif letters_list[i] == "c":
                letters_list[i] = "γ"
            elif letters_list[i] == "d":
                letters_list[i] = "δ"
            elif letters_list[i] == "e":
                letters_list[i] = "ε"
            elif letters_list[i] == "f":
                letters_list[i] = "ζ"
            elif letters_list[i] == "g":
                letters_list[i] = "η"
            elif letters_list[i] == "h":
                letters_list[i] = "θ"
            elif letters_list[i] == "i":
                letters_list[i] = "ι"
            elif letters_list[i] == "j":
                letters_list[i] = "κ"
            elif letters_list[i] == "k":
                letters_list[i] = "λ"
            elif letters_list[i] == "l":
                letters_list[i] = "μ"
            elif letters_list[i] == "m":
                letters_list[i] = "ν"
            elif letters_list[i] == "n":
                letters_list[i] = "ξ"
            elif letters_list[i] == "o":
                letters_list[i] = "ο"
            elif letters_list[i] == "p":
                letters_list[i] = "π"
            elif letters_list[i] == "q":
                letters_list[i] = "ρ"
            elif letters_list[i] == "r":
                letters_list[i] = "σ"
            elif letters_list[i] == "s":
                letters_list[i] = "τ"
            elif letters_list[i] == "t":
                letters_list[i] = "υ"
            elif letters_list[i] == "u":
                letters_list[i] = "φ"
            elif letters_list[i] == "v":
                letters_list[i] = "χ"
            elif letters_list[i] == "w":
                letters_list[i] = "ψ"
            elif letters_list[i] == "x":
                letters_list[i] = "ω"
            elif letters_list[i] == "y":
                letters_list[i] = "y"
            elif letters_list[i] == "z":
                letters_list[i] = "z"
        transformed_sentence = "".join(letters_list)
    if language == "chinese":
        letters_list = list(sentence.lower())
        for i in range(len(letters_list)):
            if letters_list[i] == "a":
                letters_list[i] = "我"
            elif letters_list[i] == "b":
                letters_list[i] = "的"
            elif letters_list[i] == "c":
                letters_list[i] = "你"
            elif letters_list[i] == "d":
                letters_list[i] = "是"
            elif letters_list[i] == "e":
                letters_list[i] = "了"
            elif letters_list[i] == "f":
                letters_list[i] = "不"
            elif letters_list[i] == "g":
                letters_list[i] = "们"
            elif letters_list[i] == "h":
                letters_list[i] = "这"
            elif letters_list[i] == "i":
                letters_list[i] = "一"
            elif letters_list[i] == "j":
                letters_list[i] = "他"
            elif letters_list[i] == "k":
                letters_list[i] = "么"
            elif letters_list[i] == "l":
                letters_list[i] = "在"
            elif letters_list[i] == "m":
                letters_list[i] = "有"
            elif letters_list[i] == "n":
                letters_list[i] = "个"
            elif letters_list[i] == "o":
                letters_list[i] = "好"
            elif letters_list[i] == "p":
                letters_list[i] = "来"
            elif letters_list[i] == "q":
                letters_list[i] = "人"
            elif letters_list[i] == "r":
                letters_list[i] = "那"
            elif letters_list[i] == "s":
                letters_list[i] = "要"
            elif letters_list[i] == "t":
                letters_list[i] = "会"
            elif letters_list[i] == "u":
                letters_list[i] = "就"
            elif letters_list[i] == "v":
                letters_list[i] = "什"
            elif letters_list[i] == "w":
                letters_list[i] = "没"
            elif letters_list[i] == "x":
                letters_list[i] = "到"
            elif letters_list[i] == "y":
                letters_list[i] = "说"
            elif letters_list[i] == "z":
                letters_list[i] = "吗"
        transformed_sentence = "".join(letters_list)
    return transformed_sentence


if __name__ == '__main__':
    #transform_all_files(r"C:\Users\RoyIlani\pythonProject\data\cc100_data_clean\cc100_en",
    #                    r"C:\Users\RoyIlani\pythonProject\data\cc100_data_clean\cc100_en_chinese_alphabet",
    #                    transform_sentences_alphabet, language="chinese")

    transform_all_gts(r"C:\Users\RoyIlani\pythonProject\data\TED_data\TED_en-zh_cn_gt\train_set",
                      "", transform_sentences_alphabet, "chinese_alphabet_transformation", language="chinese")

    transform_all_gts(r"C:\Users\RoyIlani\pythonProject\data\TED_data\TED_en-zh_cn_gt\test_set",
                      "", transform_sentences_alphabet, "chinese_alphabet_transformation", language="chinese")