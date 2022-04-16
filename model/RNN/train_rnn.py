import fasttext
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from recurrent_nn import RecurrentNN

path_to_or_train = "corpus/training/fce.train.gold.bea19.clean.or"
path_to_co_train = "corpus/training/fce.train.gold.bea19.clean.co"
path_to_or_co_train = "model/aligned.grow-diag-final-and"
path_to_word_vector = "model/RNN/cc.en.50.bin"


# ============================================ data pre-processing
class WordDataset(Dataset):
    def __init__(self, align_prev, expect):
        self.align_prev = align_prev
        self.expect = expect

    def __len__(self):
        return len(self.align_prev)

    def __getitem__(self, idx):
        return self.align_prev[idx], self.expect[idx]


def count_word(file):
    with open(file, "r") as f:
        return len(f.read().split())


def word_to_tensor(word):
    word_np = ft[word] if word is not None else np.zeros(50)
    return torch.tensor(word_np)


def build_dictionary(file):
    with open(file, "r") as f:
        keys = sorted(set(f.read().split()))
        word_dict = dict((word, index) for index, word in enumerate(keys))
        index_dict = dict((index, word) for index, word in enumerate(keys))
        return word_dict, index_dict


def output_tensor(word):
    if word in word_dictionary:
        return torch.tensor([word_dictionary[word]], dtype=torch.float32)
    else:
        return torch.tensor([-1], dtype=torch.float32)


def sentence_to_tensor(or_list, co_list, align_list):
    len_sentence = len(or_list)
    aligned_tensor = torch.zeros((len_sentence, 50), dtype=torch.float32)
    prev_tensor = torch.zeros((len_sentence, 50), dtype=torch.float32)
    expected_tensor = torch.zeros((len_sentence, 1), dtype=torch.float32)

    # create an alignment map where alignment_map[or] = co
    last_item = align_list[-1].split("-")
    alignment_map = [-1] * (max(int(last_item[0]), int(last_item[1])) + 7)
    for item in align_list:
        item_list = item.split("-")
        alignment_map[int(item_list[0])] = int(item_list[1])

    for or_pos, source in enumerate(or_list):
        co_pos = alignment_map[or_pos]
        aligned_word = co_list[co_pos] if 0 <= co_pos < len(co_list) else None
        prev_word = or_list[or_pos - 1] if 0 < or_pos < len(or_list) else None
        expected_word = source

        aligned_tensor[or_pos] = word_to_tensor(aligned_word)
        prev_tensor[or_pos] = word_to_tensor(prev_word)
        expected_tensor[or_pos] = output_tensor(expected_word)

    return aligned_tensor, prev_tensor, expected_tensor, len_sentence


def process_file(or_co, original_file, corrected_file):
    word_total_count = count_word(original_file)

    aligned_words = torch.zeros((word_total_count, 50), dtype=torch.float32)
    prev_words = torch.zeros((word_total_count, 50), dtype=torch.float32)
    expected_words = torch.zeros((word_total_count, 1), dtype=torch.long)

    with open(or_co, "r") as or_co, \
            open(original_file, "r") as or_file, \
            open(corrected_file, "r") as co_file:
        word_count = 0
        for alignment in or_co:
            # Process one sentence
            original_list = or_file.readline().split()
            correct_list = co_file.readline().split()
            alignment_list = alignment.split()
            aligned, prev, expected, len_sentence = sentence_to_tensor(original_list, correct_list, alignment_list)

            aligned_words[word_count:word_count + len_sentence, :] = aligned
            prev_words[word_count:word_count + len_sentence, :] = prev
            expected_words[word_count:word_count + len_sentence, :] = expected

            word_count += len_sentence

    return torch.cat((aligned_words, prev_words), 1), expected_words


def word_from_output(output):
    top_n, top_i = torch.topk(output, 1)
    word_i = top_i[0].item()
    return index_dictionary[word_i], word_i


ft = fasttext.load_model(path_to_word_vector)
word_dictionary, index_dictionary = build_dictionary(path_to_or_train)
n_words = len(word_dictionary)

train_input_tensor, train_expected_tensor = process_file(path_to_or_co_train, path_to_or_train, path_to_co_train)
train_word_dataset = WordDataset(train_input_tensor, torch.flatten(train_expected_tensor))
random_indices = np.random.choice(np.arange(0, len(train_input_tensor)), 50000)
train_word_subset = Subset(train_word_dataset, random_indices)

train_word_loader = DataLoader(train_word_subset, shuffle=True)

# ============================================ model training
# hyper-parameters
learning_rate = 0.005
n_epochs = 3

# create a model
rnn = RecurrentNN(100, 100, n_words)
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)


def train(input_tensor, target_tensor):
    hidden = rnn.init_hidden()
    rnn.zero_grad()
    optimizer.zero_grad()

    output, hidden = rnn(input_tensor, hidden)
    loss = loss_fn(output, target_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()


def train_loop(dataloader):
    size = len(dataloader.dataset)

    for batch, (x, y) in enumerate(dataloader):
        output, loss = train(x, y)

        if batch % 1000 == 0:
            current = batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


for t in range(n_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_word_loader)

torch.save({
    'epoch': n_epochs,
    'model_state_dict': rnn.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, "rnn_state/model_state_rnn.pth")
