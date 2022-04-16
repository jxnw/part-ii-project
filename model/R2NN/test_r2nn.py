import pandas as pd
import torch
from R2NN import R2NN, TreeNode

path_to_test_source = "corpus/dev/fce.test.gold.bea19.or"
path_to_test_translated = "evaluation/r2nn/fce.test.r2nn.translated"
path_to_pth = "evaluation/r2nn/fce.test.r2nn.translated"
path_to_pt_top = "model/pt_top"


def build_tree(span, model):
    tree_node = TreeNode()
    tree = tree_node.greedy_tree(span, model)
    return tree


def greedy_span(sentence):
    sentence_list = sentence.split()
    span = []
    start = 0
    end = 0
    while end < len(sentence_list):
        flag = False
        for length in range(5, 0, -1):
            end = start + length
            if end > len(sentence_list):
                continue
            phrases = sentence_list[start:end]
            phrase = " ".join(phrases)
            if phrase in pt_sources:
                flag = True
                span.append(phrase)
                start = end
                break
        if not flag:
            phrases = sentence_list[start:start+1]
            phrase = " ".join(phrases)
            span.append(phrase)
            start = start+1
    return span


def split_sentences(file):
    sentence_spans = []
    with open(file, "r") as f:
        for line in f:
            sentence_spans.append(greedy_span(line))
    return sentence_spans


def n_best_pair(source, targets):
    best_pair = [(source,), ("",)]
    highest_score = float('-inf')
    for target in targets:
        pair = [(source,), (target,)]
        tree = build_tree([pair], r2nn)
        _, pair_score = r2nn(tree)
        if pair_score > highest_score:
            best_pair = pair
            highest_score = pair_score
    return best_pair


def get_candidates(sentence):
    span = []
    for phrase in sentence:
        indices = [i for i, p in enumerate(pt_sources) if p == phrase]
        targets = [pt_targets[i] for i in indices]
        best_pair = n_best_pair(phrase, targets)
        span.append(best_pair)
    return span


r2nn = R2NN(21, 2)
checkpoint = torch.load(path_to_pth)
r2nn.load_state_dict(checkpoint['model_state_dict'])

pt = pd.read_csv(path_to_pt_top, sep='|')
pt_sources = pt['source'].tolist()
pt_targets = pt['target'].tolist()

sentences = split_sentences(path_to_test_source)
print("Sentences loaded")

count = 0
with open(path_to_test_translated, 'w') as translated_file:
    for s in sentences:
        count += 1
        print("Getting candidates...")
        sentence_span = get_candidates(s)
        print("Building tree...")
        sentence_tree = build_tree(sentence_span, r2nn)
        print("Tree built")
        translated_file.write(sentence_tree.target + "\n")
        if count % 10 == 0:
            print("{}/2695".format(count))
