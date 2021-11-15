# Based on PyTorch Tutorial
# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PhrasePairDataset(Dataset):
    def __init__(self, sentences, scores):
        self.sentences = sentences
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        label = self.scores[item]
        return sentence, label


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


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def main(args):
    # get one hot encodings from sentence pairs
    phrase_pair_encodings = np.zeros((2000, 200001))
    count = 0
    with open(args.phrase_pair_id) as ppt:
        phrase_ids = ppt.readline().split()
        encoding = np.zeros((1, 200001))
        for p_id in phrase_ids:
            phrase_id = int(p_id)
            if 0 <= phrase_id <= 200000:
                encoding[0, phrase_id] = 1
        phrase_pair_encodings[count] = encoding
        count += 1

    expected_score = np.random.rand(2000, 1)  # TODO: take average of features

    # Initialise the PhrasePairDataset class
    tensor_sentences = torch.Tensor(phrase_pair_encodings)
    tensor_scores = torch.Tensor(expected_score)

    phrase_pair_dataset = PhrasePairDataset(tensor_sentences, tensor_scores)
    phrase_pair_loader = DataLoader(phrase_pair_dataset)

    # Hyper-parameters
    learning_rate = 1e-3
    batch_size = 5
    epoch = 10

    # Create a model
    model = OneHiddenLayerNet(200001, 20)
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # set model to train mode
    model.train()
    for epoch_id in range(epoch):
        print(f"Epoch {epoch_id + 1}\n-------------------------------")
        train_loop(phrase_pair_loader, model, loss_fn, optimizer)


if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser()
    parser.add_argument("phrase_pair_id", help="The path to the phrase pair ids of sentence pairs.")
    args = parser.parse_args()
    main(args)
