import numpy as np
import torch
from one_hidden_layer_net import OneHiddenLayerNet


def load_model(index, save_to_hidden=False):
    path_to_pth = "model/sparse/one_hidden_layer/model_epoch{}.pth".format(index)

    model = OneHiddenLayerNet(200001, 20)
    checkpoint = torch.load(path_to_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    hidden_matrix = model.state_dict()['linear_relu_stack.0.weight'].numpy()    # 20 by 200001

    if save_to_hidden:
        path_to_hidden = "model/sparse/one_hidden_layer/hidden_epoch{}".format(index)
        hidden_avg = np.mean(hidden_matrix, axis=0)
        print("Epoch{}\n".format(index), hidden_avg)
        np.savetxt(path_to_hidden, hidden_avg, fmt="%.7f")

    hidden_matrix = hidden_matrix.T
    return hidden_matrix


path_to_ppe_sparse = "model/ppe_sparse"

ppe_matrix = load_model(10)
np.save(path_to_ppe_sparse, ppe_matrix)
