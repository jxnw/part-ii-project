import math
import torch
from torch.utils.data import Dataset, DataLoader
from R2NN import R2NN, TreeNode

path_to_phrase_used = "phrase_pair/phrases_train"


# ============================================ data pre-processing
class SentenceDataset(Dataset):
    def __init__(self, sentences, scores):
        self.sentences = sentences
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, item):
        return self.sentences[item], self.scores[item]


def process_phrases_file(file):
    # phrase_pairs[i] -> all the phrase pairs used in sentence i
    # phrase_pairs[i][j] -> jth phrase pair in sentence i
    sentences = []
    scores = []
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
                pair.append(" ".join(word_list))
            if "TRANSLATED AS:" in line:
                line = line.strip().split()
                word_list = line[2:]
                for i, p in enumerate(word_list):
                    if "|UNK|UNK|UNK" in p:
                        word_list[i] = p.replace("|UNK|UNK|UNK", "")
                pair.append(" ".join(word_list))
                assert len(pair) == 2
                sentence.append(pair)
            elif "SCORES" in line:
                # one sentence done
                if len(sentence) > 0:
                    sentences.append(sentence)
                    core = line.strip().split()[2][6:-1]
                    score_list = [float(i) for i in core.split(',')]
                    score = math.fsum(score_list)
                    scores.append(round(score, 5))
                sentence = []
    return sentences, scores


def build_tree(span, model):
    tree_node = TreeNode()
    tree = tree_node.greedy_tree(span, model)
    return tree


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (x, y) in enumerate(dataloader):
        y = y.to(torch.float32)

        # build tree
        tree = build_tree(x, model)

        # compute output score
        parent, score = model(tree)

        loss = loss_fn(score, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % batch_size == 0:
            print(score)
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch:>5d}/{size:>5d}]")


def local_loss(output, target):
    if 1 - target + output < 0:
        return torch.tensor(0., requires_grad=True)
    return 1 - target + output


filtered_sentences, filtered_scores = process_phrases_file(path_to_phrase_used)
sentence_dataset = SentenceDataset(filtered_sentences, filtered_scores)
sentence_loader = DataLoader(sentence_dataset)

# Hyper-parameters
learning_rate = 0.01
batch_size = 100
epoch = 1

r2nn = R2NN(21, 2)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(r2nn.parameters(), lr=learning_rate)

for epoch_id in range(epoch):
    print(f"Epoch {epoch_id + 1}\n-------------------------------")
    train_loop(sentence_loader, r2nn, local_loss, optimizer)

# torch.save({
#     'epoch': epoch,
#     'model_state_dict': r2nn.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict()
# }, "phrase_pair/R2NN/r2nn_state/model_state_r2nn.pth")
