# -*- coding: utf-8 -*-

import argparse
import csv
import string
import sys


class TreatmentDetector(object):
    def __init__(self, args):
        self.reader = csv.DictReader(args.posts)
        self.writer = csv.DictWriter(
            args.output, self.reader.fieldnames + ["treatments"])
        self.treatment_definitons = self.parse_treatment_definitons(args.definitons)
        self.punctuation_translator = str.maketrans('', '', string.punctuation)

    @staticmethod
    def parse_treatment_definitons(definion_file):
        treatments = set()
        for line in definion_file:
            line = line.strip()
            treatments.add(line)
        return treatments

    @staticmethod
    def normalise(text, translator):
        return text.translate(translator).lower()

    @staticmethod
    def tokenise(text):
        return [t.strip() for t in text.split(" ")]

    def detect_treatments(self):
        for post in self.reader:
            text = post["text"]
            text = self.normalise(text, self.punctuation_translator)
            tokens = set(self.tokenise(text))
            post["treatments"] = " | ".join(tokens & self.treatment_definitons)
            if post["treatments"]:
                self.writer.writerow(post)


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
