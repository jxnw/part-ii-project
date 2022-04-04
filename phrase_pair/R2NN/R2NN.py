import torch


class R2NN(torch.nn.Module):
    def __init__(self, ppe_size, rec_size):
        super(R2NN, self).__init__()
        self.input_size = ppe_size + rec_size
        self.W = torch.nn.Linear(self.input_size * 2, ppe_size)
        self.V = torch.nn.Linear(self.input_size, 1)

    def forward(self, node):
        if node.left is None or node.right is None:
            left_node = torch.zeros([self.input_size], dtype=torch.float32)
            right_node = torch.zeros([self.input_size], dtype=torch.float32)
        else:
            left_node = node.left.representation    # recurrent (x) + ppe (s)
            right_node = node.right.representation
        inp = torch.cat([left_node, right_node], 0)
        parent_ppe = torch.tanh(self.W(inp))
        parent_rec = get_rec(None)
        parent = torch.cat([parent_rec, parent_ppe], 0)
        output = self.V(parent)
        return parent, output


class TreeNode:
    def __init__(self, representation=None):
        self.representation = representation
        self.left = None
        self.right = None

    def greedy_tree(self, x, model):
        size = len(x)
        x = x.to(torch.float32)
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


def get_rec(phrases):
    # TODO: copy from train_r2nn.py
    return torch.tensor([2., 2.])


# rvnn = RecursiveNN(4, 3)
#
# s_1 = np.array([[2., 2., 2., 8., 8, 8, 8], [3, 3, 3, 2, 2, 2, 2], [4, 4, 4, 3, 3, 3, 3], [5, 5, 5, 7, 7, 7, 7]])
# t = build_tree(s_1, rvnn)
# parent_node, score = rvnn(t)
# print(parent_node, "\n", score)
