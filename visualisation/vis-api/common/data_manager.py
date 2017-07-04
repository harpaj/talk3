import csv
import logging
from datetime import datetime
import json

import pandas
import numpy as np


sent_mapping = {
    "pos": 1,
    "neu": 0,
    "neg": -1
}

factuality_weight = {
    "yes": 1,
    "no": 0.2
}

agree_weight = 0.5


class DataManager(object):
    def __init__(self, config):
        self.config = config
        logging.info("Started calculating data (this will take a bit)")
        self.df = self.prepare_initial_dataframe(self.config["paths"]["sentences"])
        self.treatment_mapping = self.prepare_treatment_definitons(
            self.config["paths"]["definitons"])
        self.treatment_summaries, self.treatment_graphs, self.treatment_detailed_data = \
            self.prepare_treatment_summaries()
        logging.info("All data calculated")

    @staticmethod
    def prepare_initial_dataframe(data_file):
        df = pandas.read_csv(
            open(data_file, 'r'),
            usecols=[
                'subforum', 'post_id', 'timestamp', 'sentence', 'treatments', 'thread_id',
                'url', 'author_id', 'sentiment', 'factuality', 'agrees'
            ],
            index_col=None,
            parse_dates=['timestamp'],
            infer_datetime_format=True
        )
        df = df.drop_duplicates()

        # calculate some initial values
        df['month'] = df['timestamp'].values.astype('<M8[M]')
        df['sentiment'] = df['sentiment'].apply(lambda s: sent_mapping[s])
        df['weight'] = df.apply(
            lambda s: factuality_weight[s['factuality']] * (1 + (s['agrees'] * agree_weight)),
            axis=1
        )
        df["weighted_sentiment"] = df["weight"] * df["sentiment"]
        return df

    @staticmethod
    def prepare_treatment_definitons(definiton_file):
        treatment_mapping = {}
        with open(definiton_file, 'r') as fh:
            reader = csv.reader(fh, skipinitialspace=True)
            for line in reader:
                treatment_mapping[line[0]] = line[1:]
        return treatment_mapping

    @staticmethod
    def calculate_score(df):
        try:
            return df["weighted_sentiment"].sum() / df["weight"].sum()
        except ZeroDivisionError:
            return 0

    def prepare_treatment_summaries(self):
        treatment_summaries = {}
        treatment_graph_data = {}
        treatment_detailed_data = {}

        # precalculate some values common to all treatments
        one_year_ago = np.datetime64(datetime.now(), 'D') - np.timedelta64(365, 'D')
        two_years_ago = one_year_ago - np.timedelta64(365, 'D')
        overall_total = len(self.df.index)
        last_year_total = len(self.df[self.df["timestamp"] > one_year_ago])
        previous_year_total = len(self.df[
            (self.df["timestamp"] > two_years_ago) & (self.df["timestamp"] < one_year_ago)])
        month_totals = self.df.groupby("month").size()

        # create groups of mentions per treatment, iterate
        for label, group in self.df.groupby("treatments"):
            data = {
                "names": self.treatment_mapping[label],
                "treatment": label
            }

            # frequency statistics
            data["total_cnt"] = len(group.index)
            data["last_year_cnt"] = len(group[
                group["timestamp"] > one_year_ago
            ])
            data["previous_year_cnt"] = len(group[
                (group["timestamp"] > two_years_ago) & (group["timestamp"] < one_year_ago)
            ])

            data["total_pcnt"] = data["total_cnt"] * 100 / overall_total
            data["last_year_pcnt"] = data["last_year_cnt"] * 100 / last_year_total
            data["previous_year_pcnt"] = data["previous_year_cnt"] * 100 / previous_year_total

            # sentiment statistics
            # # summarise the treatment sentiment on a post level
            def post_group(group):
                group["sentiment"] = self.calculate_score(group)
                group["weight"] = group["weight"].max()
                return group

            post_scores = group.groupby('post_id')[
                ['post_id', 'month', 'weight', 'weighted_sentiment', 'timestamp']
            ].apply(post_group).drop_duplicates()
            post_scores["weighted_sentiment"] = post_scores["weight"] * post_scores["sentiment"]

            # # create the overall score and last two yearly scores for the treatment,
            # # based on the scores from the posts

            data["total_score"] = self.calculate_score(post_scores)
            data["last_year_score"] = self.calculate_score(
                post_scores[post_scores["timestamp"] > one_year_ago])
            data["previous_year_score"] = self.calculate_score(post_scores[
                (post_scores["timestamp"] > two_years_ago) &
                (post_scores["timestamp"] < one_year_ago)
            ])

            treatment_summaries[label] = data

            # # summarise the posts per month
            def month_group(group):
                month = group['month'].iloc[0]

                group["score"] = self.calculate_score(group)
                group["pos_cnt"] = len(group[group["weighted_sentiment"] > 0])
                group["neu_cnt"] = len(group[group["weighted_sentiment"] == 0])
                group["neg_cnt"] = len(group[group["weighted_sentiment"] < 0])

                # these are needed for the stacked graphs
                group['pos+neu_cnt'] = group['pos_cnt'] + group['neu_cnt']
                group['all_cnt'] = group['pos+neu_cnt'] + group['neg_cnt']

                # relative counts
                group["pos_rel"] = group["pos_cnt"] * 100 / month_totals[month]
                group["neu_rel"] = group["neu_cnt"] * 100 / month_totals[month]
                group["neg_rel"] = group["neg_cnt"] * 100 / month_totals[month]
                group["pos+neu_rel"] = group["pos+neu_cnt"] * 100 / month_totals[month]
                group["all_rel"] = group["all_cnt"] * 100 / month_totals[month]
                del group["weighted_sentiment"]
                del group["weight"]
                return group

            month_groups = post_scores.groupby('month')[
                'month', 'weight', 'weighted_sentiment'
            ].apply(month_group).drop_duplicates().sort_values('month').set_index('month').reindex(
                pandas.DatetimeIndex(np.arange(
                    self.config["range"]["start"],
                    self.config["range"]["end"],
                    dtype='datetime64[M]')),
                fill_value=0
            )
            # bokeh can't read from the index, so we have to add it as a column
            month_groups['month'] = month_groups.index
            treatment_graph_data[label] = month_groups

            thread_groups = sorted(
                list(group.drop_duplicates("post_id").groupby("thread_id", sort=False)),
                key=lambda tg: len(tg[1]),
                reverse=True
            )[:5]

            detailed_data = data.copy()

            threads = []
            for thread, thread_group in thread_groups:
                sentences = thread_group.sort_values(
                    ["factuality", "agrees", "timestamp"], ascending=False
                ).head(5).sort_values("timestamp")[[
                    "url", "sentence", "sentiment", "factuality", "author_id", "timestamp",
                    "agrees"
                ]]
                sentences["timestamp"] = sentences["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
                threads.append({
                    "thread_id": thread,
                    "size": len(thread_group.index),
                    "sentences": json.loads(sentences.to_json(orient="records"))
                })
            detailed_data["threads"] = threads
            treatment_detailed_data[label] = detailed_data

            break

        return treatment_summaries, treatment_graph_data, treatment_detailed_data
