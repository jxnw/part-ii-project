import fasttext
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from RecurrentNN import RecurrentNN

path_to_or_train = "../fce.train.gold.bea19.clean.or"
path_to_co_train = "../fce.train.gold.bea19.clean.co"
path_to_or_co_train = "../aligned.grow-diag-final-and"

# path_to_or_train = "../mini_or"
# path_to_co_train = "../mini_co"
# path_to_or_co_train = "../mini_or_co"
path_to_word_vector = "cc.en.50.bin"


# ============================================ data pre-processing
class WordDataset(Dataset):
    def __init__(self, align_prev, expect):
        self.align_prev = align_prev
        self.expect = expect

    def __len__(self):
        return len(self.align_prev)

    def __getitem__(self, idx):
        return self.align_prev[idx].unsqueeze(0), self.expect[idx]


def sentence_to_np(or_list, co_list, align_list):
    len_sentence = len(or_list)

    aligned_words = np.zeros((len_sentence, 50), dtype=float)
    prev_words = np.zeros((len_sentence, 50), dtype=float)
    expected_words = np.zeros((len_sentence, 50), dtype=float)

    # convert alignment to a map where alignment_map[or] = co
    last_item = align_list[-1].split("-")
    alignment_map = [-1] * (max(int(last_item[0]), int(last_item[1])) + 5)
    for item in align_list:
        item_list = item.split("-")
        alignment_map[int(item_list[0])] = int(item_list[1])

    for or_pos, source in enumerate(or_list):
        co_pos = alignment_map[or_pos]
        aligned_word = co_list[co_pos] if 0 <= co_pos < len(co_list) else None
        prev_word = or_list[or_pos - 1] if 0 < or_pos < len(or_list) else None
        expected_word = source

        aligned_words[or_pos] = word_to_np(aligned_word)
        prev_words[or_pos] = word_to_np(prev_word)
        expected_words[or_pos] = word_to_np(expected_word)

    return aligned_words, prev_words, expected_words, len_sentence


def word_to_np(word):
    return ft[word] if word is not None else np.zeros(50)


def count_word(file):
    with open(file, "r") as f:
        return len(f.read().split())


def preprocess_file(or_co, original_file, corrected_file):
    word_total_count = count_word(original_file)
    aligned_words = np.zeros((word_total_count, 50), dtype=float)
    prev_words = np.zeros((word_total_count, 50), dtype=float)
    expected_words = np.zeros((word_total_count, 50), dtype=float)

    with open(or_co, "r") as or_co, \
            open(original_file, "r") as or_file, \
            open(corrected_file, "r") as co_file:
        word_count = 0
        for alignment in or_co:
            original = or_file.readline()
            correct = co_file.readline()

            original_list = original.split()
            correct_list = correct.split()
            alignment_list = alignment.split()

            aligned_np, prev_np, expected_np, length = sentence_to_np(original_list, correct_list, alignment_list)

            aligned_words[word_count:word_count + length, 0:50] = aligned_np
            prev_words[word_count:word_count + length, 0:50] = prev_np
            expected_words[word_count:word_count + length, 0:50] = expected_np

            word_count += length

    aligned_tensor = torch.from_numpy(aligned_words)
    prev_tensor = torch.from_numpy(prev_words)
    output_tensor = torch.from_numpy(expected_words)
    aligned_prev = torch.cat((aligned_tensor, prev_tensor), 1)

    return aligned_prev, output_tensor


# load fastText model
ft = fasttext.load_model(path_to_word_vector)

train_input_tensor, train_expected_tensor = preprocess_file(path_to_or_co_train, path_to_or_train, path_to_co_train)

train_word_dataset = WordDataset(train_input_tensor, train_expected_tensor)
train_word_loader = DataLoader(train_word_dataset, batch_size=64)


# ============================================ model training
# hyper-parameters
learning_rate = 0.005
n_epochs = 10

# create a model
model = RecurrentNN(100, 20, 50)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        output, hidden = model(x.float())
        loss = loss_fn(output, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


for t in range(n_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_word_loader, model, loss_fn, optimizer)
