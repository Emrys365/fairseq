#!/usr/bin/env python
from pathlib import Path


def prepare_lexicon(wrd_lst, delim="|"):
    lexicon = {}
    words = set()
    for lst in wrd_lst:
        with open(lst, "r") as f:
            for line in f:
                for wrd in line.strip().split():
                    if wrd:
                        words.add(wrd)
    for wrd in sorted(words):
        assert delim not in wrd, wrd
        lexicon[wrd] = list(wrd) + [delim]
    return lexicon


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wrd_lst", type=str, nargs="+", help="Paths to the text files containing words"
    )
    parser.add_argument(
        "--delim", type=str, default="|", help="Delimiter for the word boundary"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Paths to the output file for storing the lexicon",
    )
    args = parser.parse_args()

    lexicon = prepare_lexicon(args.wrd_lst, delim=args.delim)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        for wrd, lst in lexicon.items():
            f.write("{}\t{}\n".format(wrd, " ".join(lst)))
