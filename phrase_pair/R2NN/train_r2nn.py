import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from R2NN import R2NN, TreeNode


path_to_phrase_used = "phrase_pair/phrases_train"
# path_to_phrase_used = "phrase_pair/mini_what_phrases.txt"
# path_to_phrase_used = "phrase_pair/what_phrases.txt"
path_to_lm_score = "phrase_pair/lm_score_train"
path_to_ppe_matrix = "phrase_pair/ppe_matrix.npy"
path_to_pt_top = "phrase_pair/pt_top"
path_to_confidence = "phrase_pair/pt_top_id_confidence"


# ============================================ data pre-processing
class SentenceDataset(Dataset):
    def __init__(self, phrase_pair_mapping):
        self.phrase_matrix = obtain_phrase_matrix(phrase_pair_mapping)

    def __len__(self):
        return len(self.phrase_matrix)

    def __getitem__(self, item):
        return self.phrase_matrix[item], 1.


def process_file(path_to_phrases, path_to_lm_file):
    # phrase_pairs[i] -> all the phrase pairs used in sentence i
    # phrase_pairs[i][j] -> jth phrase pair in sentence i
    # phrase_pairs[i][j][0] -> source phrase
    # phrase_pairs[i][j][1] -> target phrase
    phrase_pairs = []
    lm_scores = []
    sentence_ids = []  # sentence ids which gives a forced decoding result
    with open(path_to_phrases, "r") as f:
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

    # keep lm scores of sentence_ids
    with open(path_to_lm_file, "r") as f:
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

    assert len(lm_scores) == len(phrase_pairs)
    return phrase_pairs, lm_scores


# TODO: move get_ppe, get_rec to R2NN?
def get_ppe(source_phrase, target_phrase):
    # TODO: load rnn model
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


def get_rec(source_phrase=None, target_phrase=None, sentence_id=None):
    # translation score
    t_score = 0
    mappings = pt.loc[(pt["source"] == source_phrase) & (pt["target"] == target_phrase)]
    if mappings.empty:
        # an unseen phrase pair
        pass
    else:
        # get index
        assert len(mappings.index) == 1
        mapping_id = mappings.index.tolist()[0]
        t_score = pt_id_confidence.iloc[mapping_id]["confidence"]

    # language model score
    l_score = lm_score_mapping[sentence_id]
    return torch.tensor([t_score, l_score])


def obtain_phrase_matrix(matrix):
    # Return a list of sentence span
    dataset = []
    for sentence_id, phrase_pairs in enumerate(matrix):
        sentence_span = np.zeros((len(phrase_pairs), 22))
        for pair_id, pair in enumerate(phrase_pairs):
            rec = get_rec(pair[0], pair[1], sentence_id)
            ppe = get_ppe(pair[0], pair[1])
            sentence_span[pair_id, 0:2] = rec
            sentence_span[pair_id, 2:22] = ppe
        dataset.append(sentence_span)
    print("Dataset obtained")
    return dataset


def build_tree(span, model):
    tree_node = TreeNode()
    tree = tree_node.greedy_tree(span, model)
    return tree


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    preds = np.zeros(size)
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        # build tree
        tree = build_tree(x, model)
        # compute output score
        parent, score = model(tree)
        preds[batch] = np.copy(score.data.numpy())
        loss = loss_fn(score, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % batch_size == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return preds


# load files
ppe_matrix = np.load(path_to_ppe_matrix)
pt = pd.read_csv(path_to_pt_top, sep='|')
pt_id_confidence = pd.read_csv(path_to_confidence, sep='|')

phrase_pair_mappings, lm_score_mapping = process_file(path_to_phrase_used, path_to_lm_score)

# Hyper-parameters
learning_rate = 1e-3
batch_size = 5
epoch = 10

rvnn = R2NN(20, 2)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(rvnn.parameters(), lr=learning_rate)

# sentence_dataset = SentenceDataset(phrase_pair_mappings)
# sentence_loader = DataLoader(sentence_dataset)
#
# for epoch_id in range(epoch):
#     print(f"Epoch {epoch_id + 1}\n-------------------------------")
#     confidence_score = train_loop(sentence_loader, rvnn, loss_fn, optimizer)
