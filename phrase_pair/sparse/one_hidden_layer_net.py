import torch


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
