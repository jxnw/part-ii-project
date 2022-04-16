import numpy as np

path_to_ppe_sparse = "model/ppe_sparse.npy"
path_to_ppe_rnn = "model/ppe_rnn.npy"
path_to_ppe_matrix = "model/ppe_matrix"

ppe_sparse = np.load(path_to_ppe_sparse)
ppe_rnn = np.load(path_to_ppe_rnn)

ppe_matrix = np.zeros((200001, 21))

ppe_matrix[:, 0:20] = ppe_sparse
ppe_matrix[0:200000, 20] = ppe_rnn

np.save(path_to_ppe_matrix, ppe_matrix)
