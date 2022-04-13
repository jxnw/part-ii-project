# Based on PyTorch Tutorial
# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from one_hidden_layer_net import OneHiddenLayerNet

path_to_phrase_pair_id = "../phrase_pair_id"
path_to_sentence_avg_score = "../sentence_avg_score"


# ============================================ data pre-processing
class PhrasePairDataset(Dataset):
    def __init__(self, sentences, scores):
        self.sentences = sentences
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        score = self.scores[item]
        return sentence, score


def count_line(file):
    with open(file, "r") as f:
        return len(f.readlines())


def preprocess_data(phrase_pair_id_file):
    # get one hot encodings from sentence pairs
    phrase_pair_encodings = np.zeros((training_size, 200001), dtype=bool)
    with open(phrase_pair_id_file, "r") as ppt:
        for count, line in enumerate(ppt):
            phrase_ids = line.split()
            for p_id in phrase_ids:
                phrase_pair_encodings[count, int(p_id)] = True
    return torch.BoolTensor(phrase_pair_encodings)


def load_expected(expected_file):
    expected = np.loadtxt(expected_file)
    expected = np.atleast_2d(expected).T
    return torch.Tensor(expected)


# initialise the PhrasePairDataset class
training_size = count_line(path_to_sentence_avg_score)

tensor_sentences = preprocess_data(path_to_phrase_pair_id)
tensor_scores = load_expected(path_to_sentence_avg_score)

phrase_pair_dataset = PhrasePairDataset(tensor_sentences, tensor_scores)
phrase_pair_loader = DataLoader(phrase_pair_dataset)


# ============================================ model training
# Hyper-parameters
learning_rate = 1e-3
batch_size = 100
epoch = 10

# create a model
model = OneHiddenLayerNet(200001, 20)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    preds = np.zeros(training_size)
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(x.float())
        preds[batch] = np.copy(pred.data.numpy())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % batch_size == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return preds


for epoch_id in range(epoch):
    print(f"Epoch {epoch_id + 1}\n-------------------------------")
    confidence_score = train_loop(phrase_pair_loader, model, loss_fn, optimizer)
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict()
    # }, "one_hidden_layer/model_epoch{}.pth".format(epoch_id + 1))
    # np.savetxt("one_hidden_layer/score_epoch{}".format(epoch_id + 1), confidence_score)
