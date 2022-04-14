import pandas as pd

path_to_pt_top = "phrase_pair/pt_top"
path_to_lm_total_scores = "phrase_pair/lm_total_scores"
path_to_pt_top_lm = "phrase_pair/pt_top_lm"


def process_lm_file(file):
    # keep lm scores of sentence_ids
    lm_scores = []
    with open(file, "r") as f:
        for line_id, line in enumerate(f):
            if line_id >= 200000:
                break
            lm_score = line.strip().split()[-3]
            lm_scores.append(float(lm_score))
    return lm_scores


scores = process_lm_file(path_to_lm_total_scores)
pt = pd.read_csv(path_to_pt_top, sep='|')
pt_copy = pt.copy()
pt_copy["lm"] = scores
pt_copy.to_csv(path_to_pt_top_lm, sep='|', index=False)
