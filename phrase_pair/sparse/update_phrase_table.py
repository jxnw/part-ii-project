import pandas as pd
import numpy as np

path_to_pt_rw = "phrase_pair/pt_rw"
path_to_pt_top_id = "phrase_pair/pt_top_id"

path_to_hidden = "phrase_pair/sparse/one_hidden_layer/hidden_epoch{}"
path_to_new_rw = "phrase_pair/sparse/phrase_tables/pt_rw_{}"
path_to_new_pt = "phrase_pair/sparse/phrase_tables/phrase_table_{}"

top_id = np.loadtxt(path_to_pt_top_id)

pt = pd.read_csv(path_to_pt_rw, sep='|', names=["source", "target", "scores", "alignment", "count"], quoting=3)

for i in range(1, 11):
    top_hidden = np.loadtxt(path_to_hidden.format(i))
    pt_copy = pt.copy()
    pt_copy["score5"] = [top_hidden[200000]] * len(pt)

    for j, hidden in enumerate(top_hidden):
        if j == 200000:
            continue
        pt_index = int(top_id[j])
        pt_copy.iat[pt_index, 5] = hidden

    pt_copy['scores'] = pt_copy['scores'] + " " + pt_copy['score5'].astype(str)
    pt_copy = pt_copy.drop(columns=["score5"])

    pt_copy.to_csv(path_to_new_rw.format(i), sep="|", index=False, header=False)

for i in range(1, 11):
    path_to_new_rw_i = path_to_new_rw.format(i)
    path_to_new_pt_i = path_to_new_pt.format(i)
    with open(path_to_new_rw_i, "r") as rw, \
            open(path_to_new_pt_i, "w") as pt:
        for line in rw:
            line_list = line.split("|")
            pt.write(" ||| ".join(line_list))
