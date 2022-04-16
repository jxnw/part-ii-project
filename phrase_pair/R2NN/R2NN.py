import numpy as np
import pandas as pd
import torch


class R2NN(torch.nn.Module):
    def __init__(self, ppe_size, rec_size):
        super(R2NN, self).__init__()
        self.input_size = ppe_size + rec_size
        self.W = torch.nn.Linear(self.input_size * 2, ppe_size, dtype=torch.float32)
        self.V = torch.nn.Linear(self.input_size, 1, dtype=torch.float32)
        self.relu = torch.nn.ReLU()

    def forward(self, node):
        left_node_vector = torch.zeros([self.input_size], dtype=torch.float32)
        left_node_source = ""
        left_node_target = ""
        right_node_vector = torch.zeros([self.input_size], dtype=torch.float32)
        right_node_source = ""
        right_node_target = ""

        if node.left is None and node.right is None:
            left_node_vector = node.vector
            left_node_source = node.source
            left_node_target = node.target
        if node.left is not None:
            left_node_vector = node.left.vector
            left_node_source = node.left.source
            left_node_target = node.left.target
        if node.right is not None:
            right_node_vector = node.right.vector
            right_node_source = node.right.source
            right_node_target = node.right.target

        inp_vector = torch.cat([left_node_vector, right_node_vector], 0)
        parent_ppe = torch.tanh(self.W(inp_vector))

        parent_source = left_node_source + " " + right_node_source
        parent_target = left_node_target + " " + right_node_target
        parent_rec = get_rec(parent_source, parent_target)

        parent = torch.cat([parent_rec, parent_ppe], 0)
        output = self.V(parent)
        return parent, output


class TreeNode:
    def __init__(self, vector=None, source="", target=""):
        self.vector = vector
        self.source = source
        self.target = target
        self.left = None
        self.right = None

    def greedy_tree(self, sentence_span_tuple, model):
        sentence_span = [[tup[0] for tup in pair] for pair in sentence_span_tuple]
        num_pairs = len(sentence_span)
        leafs = []
        for pair_id in range(num_pairs):
            source_phrase = sentence_span[pair_id][0]
            target_phrase = sentence_span[pair_id][1]
            pair_rec = get_rec(source_phrase, target_phrase)
            pair_ppe = get_ppe(source_phrase, target_phrase)
            pair_vector = torch.concat((pair_rec, pair_ppe), 0)

            leaf_node = TreeNode(vector=pair_vector, source=source_phrase, target=target_phrase)
            leafs.append(leaf_node)

        while len(leafs) >= 2:
            max_score = float('-inf')
            max_hypothesis = None
            max_idx = None
            for i in range(1, len(leafs)):
                hypothesis = TreeNode()
                hypothesis.left = leafs[i - 1]
                hypothesis.right = leafs[i]
                vector, score = model(hypothesis)
                hypothesis.vector = vector
                hypothesis.source = hypothesis.left.source + " " + hypothesis.right.source
                hypothesis.target = hypothesis.left.target + " " + hypothesis.right.target
                if score > max_score:
                    max_score = score
                    max_hypothesis = hypothesis
                    max_idx = i
            leafs[max_idx - 1:max_idx + 1] = [max_hypothesis]
        return leafs[0]


def get_ppe(source_phrase, target_phrase):
    # return ppe of a given phrase pair
    mappings = pt_top_lm.loc[(pt_top_lm["source"] == source_phrase) & (pt_top_lm["target"] == target_phrase)]
    if mappings.empty:
        # an unseen phrase pair
        ppe = ppe_matrix[200000]
    else:
        # get index
        assert len(mappings.index) == 1
        mapping_id = mappings.index.tolist()[0]
        ppe = ppe_matrix[mapping_id]
    return torch.tensor(ppe, dtype=torch.float32)


def get_rec(source_phrase, target_phrase):
    # translation score
    t_score = 0

    # language model score
    l_score = 0

    mappings = pt_top_lm.loc[(pt_top_lm["source"] == source_phrase) & (pt_top_lm["target"] == target_phrase)]
    if mappings.empty:
        # not in top 200000 frequent phrase pairs
        pass
    else:
        # get index
        assert len(mappings.index) == 1
        mapping_id = mappings.index.tolist()[0]
        t_score = pt_id_confidence.iloc[mapping_id]["confidence"]
        l_score = pt_top_lm.iloc[mapping_id]["lm"]

    return torch.tensor([t_score, l_score], dtype=torch.float32)


path_to_pt_top_lm = "phrase_pair/pt_top_lm"
path_to_confidence = "phrase_pair/pt_top_id_confidence"
path_to_ppe_matrix = "phrase_pair/ppe_matrix.npy"

pt_top_lm = pd.read_csv(path_to_pt_top_lm, sep='|')
pt_id_confidence = pd.read_csv(path_to_confidence, sep='|')
ppe_matrix = np.load(path_to_ppe_matrix)

