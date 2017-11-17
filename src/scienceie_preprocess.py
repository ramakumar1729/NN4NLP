"""scienceie_preprocess.py: Preprocessing to prepare features for Science IE
data.
"""
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
color_scheme='Linux', call_pdb=1)

import argparse
import glob
import itertools
import json
import os
import re
import pickle
from collections import Counter

import tqdm
import numpy as np
from pycorenlp import StanfordCoreNLP


def whitespace(match_obj):
        return " " * len(match_obj.group(0))

def preprocess_text(text):
    """Bunch of preprocessing for consistency in training/dev set."""
    text = re.sub(r"(\[\d+(-\d+)?(,\d+(-\d+)?)*\])", whitespace, text)
    text = re.sub(r"(\w+)-(\w+)", r"\1 \2", text )
    text = re.sub(r"(\w+)\.(\w+) (\w+)", r"\1. \2\3", text)
    text = re.sub(r"\s(\w+)/(\w+)\s", r" \1 \2 ", text)
    # Adhoc
    text = text.replace("W. With", "W .With")
    text = text.replace("B. Moreover", "B .Moreover")
    text = text.replace("B. Intuitively", "B .Intuitively")
    text = text.replace("pH. The", "pH .The")
    text = text.replace("UO2+x.", "UO2+x")  # Compromised.
    text = text.replace("\xa0", " ")
    return text


def load_data(save_dir):
    dataset = {}
    for type_ in ["train", "dev"]:
        data, target = [], []
        txt_paths = glob.glob(os.path.join(save_dir,
                                           "{}/data/*.txt".format(type_)))
        ann_paths = [s.replace(".txt", ".ann") for s in txt_paths]
        for tp, ap in zip(txt_paths, ann_paths):
            with open(tp, "r") as ft, open(ap, "r") as fa:
                text = ft.read().strip()  # Text is one-liner
                text = preprocess_text(text)
                ann = [l.strip("\t\n").split("\t") for l in fa.readlines()]
                for i, a in enumerate(ann):
                    if a[-1][-1] == "\xa0":
                        ann[i][-1] = ann[i][-1][:-1]
                        phrase_len = len(ann[i][-1])
                        ann[i][1] = " ".join(ann[i][1].split(" ")[:-1]+
                                             [str(int(ann[i][1].split(" ")[1])+phrase_len)])

                # Only if relations exist
                if len([a for a in ann if a[0][0] in ["*", "R"]]) > 0:
                    data.append((text, ann))
        dataset[type_] = data

    data = []
    txt_paths = glob.glob(os.path.join(save_dir,
                                       "test/data/*.txt"))
    ann_paths = [s.replace(".txt", ".ann") for s in txt_paths]

    for tp, ap in zip(txt_paths, ann_paths):
        with open(tp, "r") as ft, open(ap, "r") as fa:
            text = ft.read().strip()  # Text is one-liner
            text = preprocess_text(text)
            ann = [l.strip("\t\n").split("\t") for l in fa.readlines()]
            for i, a in enumerate(ann):
                if a[-1][-1] == "\xa0":
                    ann[i][-1] = ann[i][-1][:-1]
                    phrase_len = len(ann[i][-1])
                    ann[i][1] = " ".join(ann[i][1].split(" ")[:-1]+
                                         [str(int(ann[i][1].split(" ")[1])+phrase_len)])

            data.append((text, ann, os.path.basename(tp).split(".")[0]))

    dataset["test"] = data
    return dataset


def get_negative_pairs(ann, sent_boundaries):
    """Generate all the possible entity pairs within the same sentence."""
    positives = [tuple(a[1].replace("Arg1:", "").replace("Arg2:", "").split()[1:])
                 for a in ann if a[0][0] in ["*", "R"]]
    end2ent = {int(r[1].split()[-1]): r[0] for r in ann if r[0][0].startswith("T")}
    sent2ents = {(sbegin, sent_boundaries[i+1]): []
                 for i, sbegin in enumerate(sent_boundaries[:-1])}
    for end, ent in end2ent.items():
        for (b, e) in sent2ents:
            if b < end and end <= e:
                sent2ents[(b, e)].append(ent)

    negative_pairs = [pair for k, v in sent2ents.items()
                      for pair in itertools.combinations(v, 2)
                      if pair not in positives and pair[::-1] not in positives]

    neg_rels = [["Rn", "None Arg1:{} Arg2:{}".format(p1, p2)]
                for (p1, p2) in negative_pairs]
    return neg_rels


def extract_examples(text, ann, host, order="fix", test=False):
    """Given text and corresponding annotations, get the data examples. This
    function decides ordering strategies mentioned in [Lee et al. 2016].

    Parameters:
        text (str): Document in string form.
        ann (List): Splitted tables of annotations.
        host (pycorenlp.StanfordCoreNLP): CoreNLP client.
        order (str): ["fix", "any"]
        test (bool): True if test.
    Returns:
        example
    """
    raw_rel_annotations = [a for a in ann if a[0][0] in ["*", "R"]]
    rel_annotations = []
    # Flatten n-ary annotations into pairs
    for r in raw_rel_annotations:
        splits = r[1].split()
        if len(splits) >= 4 and splits[0] == "Synonym-of":
            pairs = itertools.combinations(splits[1:], 2)
            for p in pairs:
                rel_annotations.append([r[0],
                                        " ".join([splits[0],
                                                  p[0],
                                                  p[1]])])
        else:
            rel_annotations.append(r)

    raw_words, pos, _, sent_boundaries = extract_features(host, text)
    ann_ents, ann_rels = fix_tokenize(raw_words, ann)
    ner, ent2wordidx = fill_ent_annotation(raw_words, ann_ents)
    negative_rels = get_negative_pairs(ann, sent_boundaries)

    entity_dict = {e[0]: e[1:] for e in ann if e[0][0] not in ["*", "R"]}

    words = [w[0] for w in raw_words]

    examples = []
    failed = []
    for r in rel_annotations + negative_rels:
        rel_type, src, trg = r[1].split()
        srcid = src.replace("Arg1:", "")
        trgid = trg.replace("Arg2:", "")
        src_words = entity_dict[srcid][1].split()
        trg_words = entity_dict[trgid][1].split()
        if srcid not in ent2wordidx or trgid not in ent2wordidx:
            # CoreNLP failed to provide consistent tokenization
            if rel_type == "None":
                continue
            else:
                failed.append(r)
                continue
        src_pos = ent2wordidx[srcid]
        trg_pos = ent2wordidx[trgid]

        if src_pos < trg_pos:
            span = slice(src_pos, trg_pos+len(trg_words))
            rel_pos = ([0] * len(src_words) +
                       list(range(1, (trg_pos+len(trg_words))-(src_pos+len(src_words))+1)))
            r_rel_pos = (list(range(-(trg_pos-src_pos), 0)) +
                         [0] * len(trg_words))
            examples.append(((words[span], rel_pos, r_rel_pos,
                             ner[span], pos[span]),
                             rel_type))
            if order == "any":
                if rel_type not in ["Synonym-of", "None"]:
                    rel_type = "r_{}".format(rel_type)
                examples.append(((words[span], r_rel_pos, rel_pos,
                                 ner[span], pos[span]),
                                 rel_type))

        else:
            span = slice(trg_pos, src_pos+len(src_words))
            rel_pos = ([0] * len(trg_words) +
                       list(range(1, (src_pos+len(src_words))-(trg_pos+len(trg_words))+1)))
            r_rel_pos = (list(range(-(src_pos-trg_pos), 0)) +
                         [0] * len(src_words))
            if rel_type not in ["Synonym-of", "None"]:
                rel_type = "r_{}".format(rel_type)
            examples.append(((words[span], rel_pos, r_rel_pos,
                             ner[span], pos[span]),
                             rel_type))
            if order == "any":
                # Table 2. @paper is wrong?
                examples.append(((words[span], r_rel_pos, rel_pos,
                                 ner[span], pos[span]),
                                 rel_type))
    return examples


def fix_tokenize(words, ann):
    """check if tokenization made by StanfordCoreNLP is consistent with
    annotation. If not, merge/split expressions.
    """
    w_positions = [(w[1], w[2]) for w in words]
    begins = [i[0] for i in w_positions]
    ends = [i[1] for i in w_positions]
    rels =[a for a in ann if a[0][0] in ["*", "R"]]
    ann = [(a[0], a[1].split(), a[2].split())
           for a in ann if a[0][0] not in ["*", "R"]]

    entity_tokens = []
    for _, (_, b, _), w in ann:
        head = int(b)
        for i in w:
            entity_tokens.append((i, head, head+len(i)))
            head += len(i) + 1

    newann = []
    for a in ann:
        if int(a[1][1]) in begins:
            bidx = begins.index(int(a[1][1]))
            if int(a[1][2]) not in ends:
                # This is CoreNLP's tokenization issue
                continue
            eidx = ends.index(int(a[1][2]))
            phrase_len = len(a[2])
            tokenized_len = len(words[bidx:eidx+1])
            if phrase_len == tokenized_len:
                newann.append(a)
            elif phrase_len <= tokenized_len: # Overtokenizing
                newann.append([a[0], a[1], [w[0] for w in words[bidx:eidx+1]]])
            else:
                newann.append([a[0], a[1], [w[0] for w in words[bidx:eidx+1]]])
    return newann, rels


def fill_ent_annotation(words, ann):
    """Annotate entities in BIO scheme."""
    ner = ["O"] * len(words)
    word_begins = [w[1] for w in words]
    ent2wordidx = {}
    for eid, (type_, b, _), w in ann:
        if int(b) in word_begins:
            idx = word_begins.index(int(b))
            ent2wordidx[eid] = idx
            ner[idx] = "B-{}".format(type_)
            for i in range(1, len(w)):
                ner[idx+i] = "I-{}".format(type_)
    return ner, ent2wordidx


def extract_features(cli, doc):
    """runs Stanford CoreNLP client.

    Parameters:
        cli (pycorenlp.StanfordCoreNLP): Client.
        doc (str): A string with multiple sentences.
    Returns:
        List(dict): Features. (sentence flattened)
    """
    properties={"annotators": "tokenize,ssplit,pos,ner",
                "tokenize.options": "ptb3Dashes=false",
                "outputFormat": "json"}

    annotated = cli.annotate(doc, properties)
    texts = [s["tokens"] for s in annotated["sentences"]]
    sent_boundaries = [0]
    words, pos, ner = [], [], []

    for s in texts:
        sent_boundaries.append(s[-1]["characterOffsetEnd"])
        for w in s:
            words.append((w["word"],
                       w["characterOffsetBegin"],
                       w["characterOffsetEnd"]))
            pos.append(w["pos"])
            ner.append(w["ner"])

    return words, pos, ner, sent_boundaries


def construct_vocab(sequence, max_vocab=50000):
    """calculates and returns the lookup vocabulary.

    Parameters:
        sequence (List[object]): A flat list of objects.

    Returns:
        o2i (dict): object-to-index.
        i2o (List): index-to-object.
    """
    o2i = {"<unk>": 0, "<pad>": 1}
    i2o = ["<unk>", "<pad>"]

    freq = Counter(sequence)
    for e, _ in freq.most_common():
        if e not in o2i and len(i2o) <= max_vocab:
            o2i[e] = len(i2o)
            i2o.append(e)

    return o2i, i2o


def main(args):

    if os.path.exists(args.save_dir):
        print("Make a new save directory.")
        sys.exit(0)
    else:
        os.mkdir(args.save_dir)

    cli = StanfordCoreNLP(args.host)
    raw_dataset = load_data(args.data_dir)
    dataset = {}
    test = False
    for type_ in raw_dataset:
        examples = []
        if type_ == "test":
            test = True

        for data in tqdm.tqdm(raw_dataset[type_], ncols=80, desc=type_):
            examples += extract_examples(data[0], data[1], cli, test=test)
        dataset[type_] = examples


    # Construct vocab
    words = ([w for ex in dataset["train"] for w in ex[0][0]] +
             [w for ex in dataset["dev"] for w in ex[0][0]])
    rel_pos = ([w for ex in dataset["train"] for w in ex[0][1]+ex[0][2]] +
               [w for ex in dataset["dev"] for w in ex[0][1]+ex[0][2]])
    pos = ([w for ex in dataset["train"] for w in ex[0][4]] +
           [w for ex in dataset["dev"] for w in ex[0][4]])
    ner = ([w for ex in dataset["train"] for w in ex[0][3]] +
           [w for ex in dataset["dev"] for w in ex[0][3]])

    wvocab = construct_vocab(words)
    rvocab = construct_vocab(rel_pos)
    pvocab = construct_vocab(pos)
    nvocab = construct_vocab(ner)
    with open(os.path.join(args.save_dir, "dict.pkl"), "wb") as f:
        dicts = {
            "word": wvocab,
            "relpos": rvocab,
            "pos": pvocab,
            "ner": nvocab
        }
        pickle.dump(dicts, f)

    with open(os.path.join(args.save_dir, "data.pkl"), "wb") as f:
        pickle.dump(dataset, f)

    print("Data saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and tag data.")
    parser.add_argument("--host", type=str, help="CoreNLP host.", required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, help="Directory to the data.",
                        required=True)
    parser.add_argument("--order", type=str, help="Ordering strategy",
                        default="fix")

    args = parser.parse_args()
    main(args)
