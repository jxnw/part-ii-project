import numpy as np
import torch
from torch.utils.data import Dataset


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


learning_rate = 1e-3
model = OneHiddenLayerNet(200001, 20)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(1, 11):
    path_to_pth = "one_hidden_layer/model_epoch{}.pth".format(i)
    path_to_hidden = "one_hidden_layer/hidden_epoch{}".format(i)

    checkpoint = torch.load(path_to_pth)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(model.state_dict())

    # hidden_matrix = model.state_dict()['linear_relu_stack.0.weight'].numpy()    # 20 by 200001
    # hidden = np.mean(hidden_matrix, axis=0)
    # print("Epoch{}\n".format(i), hidden)
    # np.savetxt(path_to_hidden, hidden, fmt="%.7f")
