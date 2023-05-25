import json
from pathlib import Path


def merge_jsons(json_paths):
    spk2utt = {}
    dics = []
    keys = None
    for i, path in enumerate(json_paths):
        with open(path, "r") as f:
            dic = json.load(f)
            if i == 0:
                keys = set(dic.keys())
            if i > 0 and dic.keys() != keys:
                keys2 = set(dic.keys())
                diff_keys = keys | keys2 - keys & keys2
                raise ValueError(
                    "Different keys are found in '{}' and '{}': "
                    "{}".format(json_paths[0], path, diff_keys)
                )
            dics.append(dic)

    for sid in keys:
        uids = None
        for i, lst in enumerate([dic[sid] for dic in dics]):
            if i == 0:
                uids = tuple(tup[0] for tup in lst)
            elif tuple(tup[0] for tup in lst) != uids:
                raise ValueError(
                    "Different uids are found for sid={} in '{}' and '{}':\n"
                    "{}\n{}".format(
                        sid, json_paths[0], path, uids, tuple(tup[0] for tup in lst)
                    )
                )
        spk2utt[sid] = [
            (uid, *[dic[sid][i][1] for dic in dics]) for i, uid in enumerate(uids)
        ]
    return spk2utt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_paths",
        type=str,
        nargs="+",
        help="Paths to Librispeech subsets",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="spk2utt_tse.json",
        help="Path to the output spk2utt json file",
    )
    args = parser.parse_args()

    spk2utt = merge_jsons(args.json_paths)

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        json.dump(spk2utt, f)
