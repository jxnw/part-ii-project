import numpy as np

path_to_ppe_sparse = "phrase_pair/ppe_sparse.npy"
path_to_ppe_rnn = "phrase_pair/ppe_rnn.npy"
path_to_ppe_matrix = "phrase_pair/ppe_matrix"

ppe_sparse = np.load(path_to_ppe_sparse)
ppe_rnn = np.load(path_to_ppe_rnn)

ppe_matrix = np.zeros((200001, 21))

ppe_matrix[:, 0:20] = ppe_sparse
ppe_matrix[0:200000, 20] = ppe_rnn

np.save(path_to_ppe_matrix, ppe_matrix)
