# Based on PyTorch Tutorial
# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

path_to_phrase_pair_id = "../phrase_pair_id"
path_to_sentence_avg_score = "../sentence_avg_score"


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


class OneHiddenLayerNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(OneHiddenLayerNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, 1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output


training_size = 28327

# get one hot encodings from sentence pairs
phrase_pair_encodings = np.zeros((training_size, 200001), dtype=bool)
with open(path_to_phrase_pair_id, "r") as ppt:
    for count, line in enumerate(ppt):
        phrase_ids = line.split()
        for p_id in phrase_ids:
            phrase_pair_encodings[count, int(p_id)] = True

# get average feature score
expected_score = np.loadtxt(path_to_sentence_avg_score)
expected_score = np.atleast_2d(expected_score).T

# Initialise the PhrasePairDataset class
tensor_sentences = torch.BoolTensor(phrase_pair_encodings)
tensor_scores = torch.Tensor(expected_score)

phrase_pair_dataset = PhrasePairDataset(tensor_sentences, tensor_scores)
phrase_pair_loader = DataLoader(phrase_pair_dataset)

# Hyper-parameters
learning_rate = 1e-3
batch_size = 100
epoch = 10

# Create a model
model = OneHiddenLayerNet(200001, 20)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for param in model.state_dict():
    print(param)


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


# set model to train mode
model.train()
for epoch_id in range(epoch):
    print(f"Epoch {epoch_id + 1}\n-------------------------------")
    confidence_score = train_loop(phrase_pair_loader, model, loss_fn, optimizer)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, "one_hidden_layer/model_epoch{}.pth".format(epoch_id + 1))
    np.savetxt("one_hidden_layer/score_epoch{}".format(epoch_id + 1), confidence_score)
