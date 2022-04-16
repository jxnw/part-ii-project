import torch


class RecurrentNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecurrentNN, self).__init__()
        self.hidden_size = hidden_size

        self.U = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.V = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inp, hidden):
        concat_inp = torch.cat((inp, hidden), 1)
        hidden = self.U(concat_inp)
        output = self.V(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
