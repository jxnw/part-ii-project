import numpy as np
import torch


class RecursiveNN(torch.nn.Module):
    def __init__(self, input_size):
        super(RecursiveNN, self).__init__()
        self.W = torch.nn.Linear(input_size * 2, input_size)
        self.V = torch.nn.Linear(input_size, 1)

    def forward(self, node):
        left_node = node.left.representation    # recurrent (x) + ppe (s)
        right_node = node.right.representation
        inp = torch.cat([left_node, right_node], 0)
        parent = torch.tanh(self.W(inp))
        output = self.V(parent)
        return parent, output


class TreeNode:
    def __init__(self, representation=None):
        self.representation = representation
        self.left = None
        self.right = None

    def greedy_tree(self, x, model):
        size = len(x)
        x = torch.tensor(x, dtype=torch.float32)
        leafs = [TreeNode(x[i, :]) for i in range(size)]
        while len(leafs) >= 2:
            max_score = float('-inf')
            max_hypothesis = None
            max_idx = None
            for i in range(1, len(leafs)):
                hypothesis = TreeNode()
                hypothesis.left = leafs[i - 1]
                hypothesis.right = leafs[i]
                representation, score = model(hypothesis)
                hypothesis.representation = representation
                if score > max_score:
                    max_score = score
                    max_hypothesis = hypothesis
                    max_idx = i
            leafs[max_idx - 1:max_idx + 1] = [max_hypothesis]
        return leafs[0]


rvnn = RecursiveNN(4)

s_1 = np.array([[8., 8, 8, 8], [2, 2, 2, 2], [3, 3, 3, 3], [7, 7, 7, 7]])
tree_node = TreeNode()
ans = tree_node.greedy_tree(s_1, rvnn)

parent_node, score = rvnn(ans)
print(parent_node, score)
