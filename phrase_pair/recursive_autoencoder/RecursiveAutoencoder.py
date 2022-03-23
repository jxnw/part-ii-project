import torch


class RecursiveAutoencoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecursiveAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.output_size),
            torch.nn.ReLU()
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output

# TODO: for each phrase pair in phrase table:
#           find embedding for s1, s2
#           encode -> decode -> output s1', s2'
#           loss = ...
