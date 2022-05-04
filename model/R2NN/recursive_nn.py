import numpy as np
import torch


class RecursiveNN(torch.nn.Module):
    def __init__(self, input_size):
        super(RecursiveNN, self).__init__()
        self.W = torch.nn.Linear(input_size * 2, input_size)
        self.V = torch.nn.Linear(input_size, 1)

    def forward(self, node):
        left_node = node.left.representation
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

s = np.array([[8., 8, 8, 8], [2, 2, 2, 2], [3, 3, 3, 3], [7, 7, 7, 7]])
tree_node = TreeNode()
tree = tree_node.greedy_tree(s, rvnn)

parent_node, output_score = rvnn(tree)
print(parent_node, output_score)


def dfs(root):
    if root is not None:
        dfs(root.left)
        print(root.representation)
        dfs(root.right)


def bfs(root):
    q = [(root, 0, -1)]
    current_id = 1
    while len(q) > 0:
        size = len(q)
        for i in range(size):
            t = q[i][0]
            t_id = q[i][1]
            parent_id = q[i][2]
            print(t_id, "\t", parent_id, "\t", t.representation)
            if t.left is not None:
                q.append((t.left, current_id, t_id))
                current_id += 1
            if t.right is not None:
                q.append((t.right, current_id, t_id))
                current_id += 1
        q = q[size:]


bfs(tree)
