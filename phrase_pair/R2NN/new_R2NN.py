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


def get_rec(pt_top, pt_id_confidence, lm_score_mapping, source_phrase=None, target_phrase=None, sentence_id=None):
    # translation score
    t_score = 0
    mappings = pt_top.loc[(pt_top["source"] == source_phrase) & (pt_top["target"] == target_phrase)]
    if mappings.empty:
        # not in top 200000 frequent phrase pairs
        pass
    else:
        # get index
        assert len(mappings.index) == 1
        mapping_id = mappings.index.tolist()[0]
        t_score = pt_id_confidence.iloc[mapping_id]["confidence"]

    # language model score
    l_score = lm_score_mapping[sentence_id]
    return torch.tensor([t_score, l_score])


model = R2NN(21, 2)
