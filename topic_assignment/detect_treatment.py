# -*- coding: utf-8 -*-

import argparse
import csv
import string
import sys
import itertools as it
from collections import Counter
from pprint import pprint

import nltk


class TreatmentDetector(object):
    def __init__(self, args):
        self.reader = csv.DictReader(args.posts)
        self.writer = csv.DictWriter(
            args.output, self.reader.fieldnames + ["sentence", "after", "third", "treatments"])
        self.writer.writeheader()
        self.treatment_set, self.treatment_mapping = self.parse_treatment_definitons(args.definitons)
        self.punctuation_translator = str.maketrans('', '', string.punctuation)
        self.sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.found_treatments = Counter()

    @staticmethod
    def parse_treatment_definitons(definion_file):
        treatment_set = set()
        treatment_mapping = dict()
        for line in definion_file:
            line = line.strip()
            treatments = line.split(',')
            name = treatments[0]
            for treatment in treatments:
                treatment = treatment.strip().lower()
                treatment_set.add(treatment)
                treatment_mapping[treatment] = name
        return treatment_set, treatment_mapping

    def normalise(self, text):
        return text.translate(self.punctuation_translator).replace("\n", " ").lower()

    @staticmethod
    def tokenise(text):
        return [t.strip() for t in text.split(" ")]

    def sentence_splitting(self, text):
        return self.sentence_detector.tokenize(text)

    # adopted from https://stackoverflow.com/a/7004905
    @staticmethod
    def window_sliding(iterable):
        gens = (
            it.chain(it.repeat(None, 2 - i), iterable, it.repeat(None, i))
            for i, gen in enumerate(it.tee(iterable, 3)))
        return zip(*gens)

    def detect_treatments(self):
        for post in self.reader:
            for sentence, after, third in self.window_sliding(
                self.sentence_splitting(post["text"])
            ):
                if not sentence:
                    continue
                tokens = set(self.tokenise(self.normalise(sentence)))
                detected_treatments = tokens & self.treatment_set
                mapped_treatments = " | ".join([
                    "{} ({})".format(t, self.treatment_mapping[t])
                    for t in detected_treatments
                ])
                for t in detected_treatments:
                    self.found_treatments[self.treatment_mapping[t]] += 1
                if detected_treatments:
                    outp = post.copy()
                    outp.pop("text")
                    outp["sentence"] = sentence.replace("\n", " ")
                    outp["after"] = after.replace("\n", " ") if after else None
                    outp["third"] = third.replace("\n", " ") if third else None
                    outp["treatments"] = mapped_treatments
                    self.writer.writerow(outp)
        pprint(self.found_treatments, sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--posts', '-p', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--definitons', '-d', type=argparse.FileType('r'), default='data/treatment_definitons.txt')
    parser.add_argument('--output', '-o', type=argparse.FileType('w'), default=sys.stdout)
    return parser.parse_args()


def main():
    args = parse_args()
    td = TreatmentDetector(args)
    td.detect_treatments()


if __name__ == "__main__":
    main()
