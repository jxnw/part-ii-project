import fasttext
import numpy as np
import pandas as pd
import torch
from recurrent_nn import RecurrentNN


path_to_or_train = "../fce.train.gold.bea19.clean.or"
path_to_pth = "rnn_state/model_state_rnn.pth"
path_to_pt_top = "../pt_top"
path_to_word_vector = "cc.en.50.bin"
path_to_ppe_rnn = "../ppe_rnn"


def build_dictionary(file):
    with open(file, "r") as f:
        keys = sorted(set(f.read().split()))
        word_dict = dict((word, index) for index, word in enumerate(keys))
        index_dict = dict((index, word) for index, word in enumerate(keys))
        return word_dict, index_dict


def word_to_tensor(word):
    word_np = ft[word] if word is not None else np.zeros(50)
    return torch.tensor(word_np, dtype=torch.float32)


def word_prob_from_output(output, word):
    if word in word_dictionary:
        word_index = word_dictionary[word]
        prob = -output[0][word_index]
    else:
        prob = 0
    return prob


word_dictionary, index_dictionary = build_dictionary(path_to_or_train)
n_word = len(word_dictionary)
print("Number of unique words in source file: ", n_word)

model = RecurrentNN(100, 100, n_word)
hidden = model.init_hidden()
checkpoint = torch.load(path_to_pth)
model.load_state_dict(checkpoint['model_state_dict'])


ft = fasttext.load_model(path_to_word_vector)
pt = pd.read_csv(path_to_pt_top, sep='|')

# for each of the top 200000 phrase pairs, calculate score
source_phrase_list = pt['source'].tolist()
target_phrase_list = pt['target'].tolist()
alignment_list = pt['alignment'].tolist()
assert len(source_phrase_list) == len(target_phrase_list) == len(alignment_list) == 200000

ppe_rnn = np.zeros(200000)

for p_id in range(len(source_phrase_list)):
    score = 0

    source_list = source_phrase_list[p_id].split()
    target_list = target_phrase_list[p_id].split()
    align_list = alignment_list[p_id].split()

    # create an alignment map where alignment_map[or] = co
    last_item = align_list[-1].split("-")
    align_map = [-1] * (max(int(last_item[0]), int(last_item[1])) + 7)
    for item in align_list:
        item_list = item.split("-")
        align_map[int(item_list[0])] = int(item_list[1])

    for or_pos, source in enumerate(source_list):
        # print(or_pos, source)
        co_pos = align_map[or_pos]
        aligned_word = target_list[co_pos] if 0 <= co_pos < len(target_list) else None
        prev_word = source_list[or_pos - 1] if 0 < or_pos < len(source_list) else None

        aligned_tensor = word_to_tensor(aligned_word).unsqueeze(0)
        prev_tensor = word_to_tensor(prev_word).unsqueeze(0)

        align_prev = torch.cat((aligned_tensor, prev_tensor), 1)
        out, _ = model(align_prev, hidden)
        prob = word_prob_from_output(out, source)
        score += prob.item()

    ppe_rnn[p_id] = score

np.save(path_to_ppe_rnn, ppe_rnn)
