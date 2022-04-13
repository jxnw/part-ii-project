import numpy as np
import pandas as pd
import torch


path_to_lm_file = "phrase_pair/lm_score_train"
path_to_phrase_used = "phrase_pair/phrases_train"


def process_phrases_file(file):
    # phrase_pairs[i] -> all the phrase pairs used in sentence i
    # phrase_pairs[i][j] -> jth phrase pair in sentence i
    phrase_pairs = []
    sentence_ids = []  # sentence ids which gives a forced decoding result
    with open(file, "r") as f:
        sentence = []
        sentence_id = -1
        for line in f:
            if "TRANSLATION HYPOTHESIS DETAILS:" in line:
                sentence_id += 1
            if "SOURCE: [" in line:
                pair = []
                line = line.strip().split()
                word_list = line[2:]
                pair.append(" ".join(word_list) + " ")
            if "TRANSLATED AS:" in line:
                line = line.strip().split()
                word_list = line[2:]
                for i, p in enumerate(word_list):
                    if "|UNK|UNK|UNK" in p:
                        word_list[i] = p.replace("|UNK|UNK|UNK", "")
                pair.append(" " + " ".join(word_list) + " ")
                assert len(pair) == 2
                sentence.append(pair)
            elif "SCORES" in line:
                # one sentence done
                if len(sentence) > 0:
                    phrase_pairs.append(sentence)
                sentence_ids.append(sentence_id)
                sentence = []
    return sentence_ids, phrase_pairs


def process_lm_file(file, sentence_ids):
    # keep lm scores of sentence_ids
    lm_scores = []
    with open(file, "r") as f:
        pointer = 0
        target_id = sentence_ids[pointer]
        for line_id, line in enumerate(f):
            if line_id == target_id:
                lm_score = line.strip().split()[-3]
                lm_scores.append(float(lm_score))
                pointer += 1
                if pointer >= len(sentence_ids):
                    break
                target_id = sentence_ids[pointer]
    return lm_scores


filtered_ids, filtered_phrases = process_phrases_file(path_to_phrase_used)
filtered_lm = process_lm_file(path_to_lm_file, filtered_ids)
print(len(filtered_phrases))
