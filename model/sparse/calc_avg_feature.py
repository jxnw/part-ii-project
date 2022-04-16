import pandas as pd

path_to_pt_top = "model/pt_top"
path_to_confidence = "model/pt_top_id_confidence"
path_to_phrase_pair_id = "model/phrase_pair_id"
path_to_avg_score = "model/sentence_avg_score"

# ==================================================== Calculate average feature for each phrase pair
pt = pd.read_csv(path_to_pt_top, sep='|')
pt["confidence"] = pt["scores"].map(lambda scores: round(sum([float(i) for i in scores.split()]) / 4, 6))
pt_id_confidence = pt.drop(["source", "target", "scores", "alignment"], axis=1)
pt_id_confidence.to_csv(path_to_confidence, sep='|', index=False)

# ==================================================== Calculate average score for each sentence pair
pt_id_confidence = pd.read_csv(path_to_confidence, sep='|')

with open(path_to_phrase_pair_id, "r") as ppi, \
        open(path_to_avg_score, "w") as output:
    for line in ppi:
        phrase_ids = line.split()
        total_score = 0
        num_pid = 0
        for p_id in phrase_ids:
            if int(p_id) < 200000:
                confidence = pt_id_confidence.iloc[int(p_id)]["confidence"]
                total_score += confidence
                num_pid += 1
        avg = round(total_score / num_pid, 6) if num_pid != 0 else 0.5
        output.write(str(avg) + "\n")
