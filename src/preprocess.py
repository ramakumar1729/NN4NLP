import argparse
import os
import re
import sys

import nltk
import numpy as np

pattern = re.compile(r"[,\(\)]")


def convert_n_tokenize(s: str):
    """Convert <> tag into underscore representation like _entity1_ . This is to
    not confuse nltk punkt tokenizer, which is needed for tokenizing punctuation
    """
    s = (s.strip()
         .replace("<", " _")
         .replace(">", "_ ")
         .replace("  ", " "))

    s = nltk.word_tokenize(s)
    s.pop(s.index("_/P_"))
    s.pop(s.index("_/C_"))
    return s

def main(args):

    # Make a save directory
    if os.path.exists(args.save_dir):
        print("Make a new save directory.")
        sys.exit(1)
    else:
        os.mkdir(args.save_dir)

    with open(args.trainX, "r") as f:
        xraw = f.read().strip().split("\n")
        x = [convert_n_tokenize(s) for s in xraw]

    with open(args.trainY, "r") as f:
        yraw = f.read().strip().split("\n")
        y = [re.split(pattern, l)[:-1] for l in yraw]

    assert len(x) == len(y)

    # Split dataset
    num_dev_examples = len(x) // args.dev_ratio
    x, y = np.array(x), np.array(y)
    idcs = np.arange(len(x))
    np.random.shuffle(idcs)

    train_x, train_y = x[idcs[num_dev_examples+1:]], y[idcs[num_dev_examples+1:]]
    dev_x, dev_y = x[idcs[:num_dev_examples]], y[idcs[:num_dev_examples]]
    print("Train: {}, Dev: {}".format(len(train_x), len(dev_x)))

    with open(os.path.join(args.save_dir, "train.txt"), "w") as f:
        for i, j in zip(train_x, train_y):
            print("{}\t{}".format(" ".join(i), " ".join(j)), file=f)

    with open(os.path.join(args.save_dir, "dev.txt"), "w") as f:
        for i, j in zip(dev_x, dev_y):
            print("{}\t{}".format(" ".join(i), " ".join(j)), file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess the constructed data.")
    parser.add_argument("--trainX", type=str, required=True)
    parser.add_argument("--trainY", type=str, required=True)
    parser.add_argument("--dev-ratio", type=float, default=10,
                        help="How much percentage of data to use as dev.")
    parser.add_argument("--save-dir", type=str)
    args = parser.parse_args()
    main(args)
