import argparse
import pandas as pd


def main(args):
    # ===================================  Read top 200,000 phrase pairs
    pt = pd.read_csv(args.phrase_table, sep='|')

    # ===================================  Read training files and alignment file
    train_or = open(args.source, "r")
    train_co = open(args.target, "r")
    or_co = open(args.alignment, "r")

    output = open(args.out, "w")

    for alignment in or_co.readlines():
        original = train_or.readline()
        correct = train_co.readline()

        original_list = original.split()
        correct_list = correct.split()
        alignment_list = alignment.split()

        phrase_pair_indices = []

        # convert alignment to a map where map[or] = co
        last_item = alignment_list[-1].split("-")
        alignment_map = [-1] * (max(int(last_item[0]), int(last_item[1])) + 1)
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
                phrase_pair_indices.append("200000")
                continue
            else:
                # get index
                assert len(mappings.index) == 1
                mapping_id = mappings.index.tolist()[0]
                phrase_pair_indices.append(str(mapping_id))

        output.write(" ".join(phrase_pair_indices) + "\n")

    train_or.close()
    train_co.close()
    or_co.close()
    output.close()


if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser()
    parser.add_argument("phrase_table", help="The path to the (top) phrase table.")
    parser.add_argument("-source", help="The path to the training source file.", required=True)
    parser.add_argument("-target", help="The path to the training target file.", required=True)
    parser.add_argument("-alignment", help="The path to the alignment file.", required=True)
    parser.add_argument("-target", help="The path to the training target file.", required=True)
    parser.add_argument("-out", help="A path to where we save the phrase pair ids of each sentence pair.", required=True)
    args = parser.parse_args()
    main(args)