# -*- coding: utf-8 -*-

import argparse
import sys

import pandas


class Visualisor(object):
    def __init__(self, args):
        self.df = pandas.read_csv(
            args.input,
            usecols=['subforum', 'post_id', 'timestamp', 'sentence', 'treatments'],
            index_col='post_id',
            parse_dates=['timestamp'],
            infer_datetime_format=True
        )
        print(self.df.head())
        print(len(self.df.index))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--output', '-o', type=argparse.FileType('w'), default=sys.stdout)
    return parser.parse_args()


def main():
    args = parse_args()
    vs = Visualisor(args)
    vs.visualise()


if __name__ == "__main__":
    main()
