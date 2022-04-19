import pandas as pd

path_to_pt_top = "model/pt_top"
path_to_or = "corpus/training/fce.train.gold.bea19.or"
path_to_co = "corpus/training/fce.train.gold.bea19.co"
path_to_or_co = "model/aligned.grow-diag-final-and"
path_to_output = "model/phrase_pair_id"

# ===================================  Read top 200,000 phrase pairs
pt = pd.read_csv(path_to_pt_top, sep='|')

# ===================================  Read training files and alignment file
with open(path_to_or, "r") as train_or, \
        open(path_to_co, "r") as train_co, \
        open(path_to_or_co, "r") as or_co, \
        open(path_to_output, "w") as output:
    for alignment in or_co:
        original = train_or.readline()
        correct = train_co.readline()

        original_list = original.split()
        correct_list = correct.split()
        alignment_list = alignment.split()

        phrase_pair_indices = []
        unseen_flag = False

        # convert alignment to a map where map[or] = co
        last_item = alignment_list[-1].split("-")
        alignment_map = [-1] * (max(int(last_item[0]), int(last_item[1])) + 5)
        for item in alignment_list:
            item_list = item.split("-")
            alignment_map[int(item_list[0])] = int(item_list[1])

        # find all phrase pairs in source of length < 7
        index = [i for i in range(len(original_list))]
        source_phrase_ids = []
        for start in range(len(original_list)):
            for length in range(7):
                end = start + length + 1
                if end > len(original_list):
                    break
                source_phrase_ids.append(index[start:end])

        # find all target phrase pairs for each source phrase pair
        target_phrase_ids = []
        for source_phrase_id in source_phrase_ids:
            target_phrase_id = []
            for or_pos in source_phrase_id:
                co_pos = alignment_map[or_pos]
                if co_pos not in target_phrase_id:
                    target_phrase_id.append(co_pos)
            target_phrase_ids.append(target_phrase_id)

        for source_phrase_id, target_phrase_id in zip(source_phrase_ids, target_phrase_ids):
            # convert source_phrase_id and target_phrase_id to phrases
            source_phrase_list = [original_list[x] for x in source_phrase_id if x != -1]
            target_phrase_list = [correct_list[x] for x in target_phrase_id if x != -1]
            source_phrase = " ".join(source_phrase_list) + " "
            target_phrase = " " + " ".join(target_phrase_list) + " "

            # look up in the phrase table
            mappings = pt.loc[(pt["source"] == source_phrase) & (pt["target"] == target_phrase)]
            if mappings.empty:
                # the sentence pair contains an unseen phrase pair
                if not unseen_flag:
                    unseen_flag = True
                    phrase_pair_indices.append("200000")
            else:
                # get index
                assert len(mappings.index) == 1
                mapping_id = mappings.index.tolist()[0]
                phrase_pair_indices.append(str(mapping_id))
        output.write(" ".join(phrase_pair_indices) + "\n")
