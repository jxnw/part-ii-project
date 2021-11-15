import argparse
import pandas as pd


def main(args):
    # ===================================  Reformat phrase table to be read by pandas
    with open(args.phrase_table, "r") as f:
        out = open("pt-rw", "w")
        for rows in f:
            row = rows.split(" ||| ")
            out.write(" | ".join(row[:-1]) + "\n")
        out.close()

    # ===================================  Find top 200,000 frequent phrase pairs
    pt = pd.read_csv(args.reformat_pt, sep='|', names=["source", "target", "scores", "alignment", "count"])
    pt[["count_co", "count_or", "count_or_co"]] = pt["count"].str.split(expand=True)
    pt["count_co"] = pd.to_numeric(pt["count_co"])
    pt["count_or"] = pd.to_numeric(pt["count_or"])
    pt["count_or_co"] = pd.to_numeric(pt["count_or_co"])
    pt_sort = pt.sort_values(["count_or_co", "count_co", "count_or"], ascending=False)
    most_freq_pt = pt_sort.head(200000)
    most_freq_pt = most_freq_pt.reset_index()
    most_freq_pt = most_freq_pt.drop(columns=["count", "count_or_co", "count_co", "count_or"])

    # ===================================  Write to file
    most_freq_pt.to_csv(args.out, sep='|', index=False)


if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser()
    parser.add_argument("phrase_table", help="The path to the phrase table.")
    parser.add_argument("-reformat_pt", help="A path to where we save the reformatted phrase table.", required=True)
    parser.add_argument("-out", help="A path to where we save the phrase table of top frequent pairs.", required=True)
    args = parser.parse_args()
    main(args)
