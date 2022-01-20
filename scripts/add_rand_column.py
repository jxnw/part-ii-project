import argparse
import random


def main(args):
    phrase_table = open(args.phrase_table).read().strip().split("\n")
    out = open(args.out, "w")

    for rows in phrase_table:
        # Add a column to phrase table
        row = rows.split(" ||| ")

        # Update score
        score = row[2].split()
        score.append(str(round(random.random(), 3)))
        row[2] = " ".join(score)
        out.write(" ||| ".join(row) + "\n")


if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser()
    parser.add_argument("phrase_table", help="The path to phrase table.")
    parser.add_argument("-out", help="A path to where we save the updated phrase table.", required=True)
    args = parser.parse_args()
    main(args)
