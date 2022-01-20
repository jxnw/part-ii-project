import pandas as pd

path_to_pt = "../phrase_table"
path_to_pt_rw = "../pt_rw"  # change ||| to |
path_to_pt_top = "../pt_top"  # top 200000 phrase pairs
path_to_top_id = "../pt_top_id"

# ===================================  Reformat phrase table to be read by pandas
with open(path_to_pt, "r") as f, open(path_to_pt_rw, "w") as out:
    for rows in f:
        row = rows.split(" ||| ")
        out.write(" | ".join(row[:-1])+"\n")

# ===================================  Find top 200,000 frequent phrase pairs
pt = pd.read_csv(path_to_pt_rw, sep='|', names=["source", "target", "scores", "alignment", "count"], quoting=3)
pt[["count_co", "count_or", "count_or_co"]] = pt["count"].str.split(expand=True)
pt["count_co"] = pd.to_numeric(pt["count_co"])
pt["count_or"] = pd.to_numeric(pt["count_or"])
pt["count_or_co"] = pd.to_numeric(pt["count_or_co"])
pt_sort = pt.sort_values(["count_or_co", "count_co", "count_or"], ascending=False)
most_freq_pt = pt_sort.head(200000)
most_freq_pt = most_freq_pt.reset_index()
most_freq_pt = most_freq_pt.drop(columns=["count", "count_or_co", "count_co", "count_or"])

pt_top_id = most_freq_pt.drop(columns=['source', 'target', 'scores', 'alignment'])

# ===================================  Write to file
most_freq_pt.to_csv(path_to_pt_top, sep='|', index=False)
pt_top_id.to_csv(path_to_top_id, sep='|')
