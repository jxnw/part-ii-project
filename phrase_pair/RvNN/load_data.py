import numpy as np
import pandas as pd
import torch
from RecursiveNN import RecursiveNN, TreeNode


path_to_phrase_used = "phrase_pair/mini_what_phrases.txt"
path_to_ppe_matrix = "phrase_pair/ppe_matrix.npy"
path_to_pt_top = "phrase_pair/pt_top"


def obtain_phrase_pairs(path):
    # Return a matrix where
    # matrix[i] -> all the phrase pairs used in sentence i
    phrase_pair_mappings = []
    with open(path, "r") as f:
        sentence = []
        for line in f:
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
                    phrase_pair_mappings.append(sentence)
                sentence = []
    return phrase_pair_mappings


def get_ppe(source_phrase, target_phrase):
    # return ppe of a given phrase pair
    mappings = pt.loc[(pt["source"] == source_phrase) & (pt["target"] == target_phrase)]
    if mappings.empty:
        # an unseen phrase pair
        return ppe_matrix[200000]
    else:
        # get index
        assert len(mappings.index) == 1
        mapping_id = mappings.index.tolist()[0]
        return ppe_matrix[mapping_id]


def get_rec(phrases):
    # TODO: based on phrases, obtain recurrent vector
    return torch.tensor([2., 2., 2.])


ppe_matrix = np.load(path_to_ppe_matrix)
pt = pd.read_csv(path_to_pt_top, sep='|')

phrase_pairs_used = obtain_phrase_pairs(path_to_phrase_used)

rvnn = RecursiveNN(20, 3)

for sentence_id, phrase_pairs in enumerate(phrase_pairs_used):
    # TODO: construct tree based on phrase_pairs
    leaf_nodes = np.zeros((len(phrase_pairs), 23))
    for pair_id, pair in enumerate(phrase_pairs):
        ppe = get_ppe(pair[0], pair[1])
        rec = get_rec(pair[0])
        leaf_nodes[pair_id, 0:20] = ppe
        leaf_nodes[pair_id, 20:23] = rec

    tree_node = TreeNode()
    tree = tree_node.greedy_tree(leaf_nodes, rvnn)

    parent_node, score = rvnn(tree)
    print(parent_node, score)

