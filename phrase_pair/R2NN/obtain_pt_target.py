import pandas as pd

path_to_pt_top = "phrase_pair/pt_top"
path_to_target = "phrase_pair/pt_top_target"

pt = pd.read_csv(path_to_pt_top, sep='|')
pt["target"].to_csv(path_to_target, index=False, header=False)
