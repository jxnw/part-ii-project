import argparse
import glob


def main(args):
    file_names = glob.glob(args.source)
    out = open(args.out, 'w')
    for file_name in file_names:
        with open(file_name, 'r') as file:
            for line in file:
                out.write(line)


if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser()
    parser.add_argument("-source", help="A path to the folder of files to be concatenated", required=True)
    parser.add_argument("-out", help="A path to where we save the output concatenated file.", required=True)
    args = parser.parse_args()
    main(args)

