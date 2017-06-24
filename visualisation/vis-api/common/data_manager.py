import csv
import logging
from datetime import datetime

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
    def __init__(self):
        logging.info("Started calculating data (this will take a bit)")
        self.df = self.prepare_initial_dataframe()
        self.data_month_crosstab = self.data_month_crosstab()
        self.treatment_mapping = self.prepare_treatment_definitons()
        self.treatment_summaries, self.treatment_graphs = self.prepare_treatment_summaries()
        logging.info("All data calculated")

    @staticmethod
    def prepare_initial_dataframe():
        df = pandas.read_csv(
            open("/home/johannes/talk3/data/treatment_detected_linewise.csv", 'r'),
            usecols=[
                'subforum', 'post_id', 'timestamp', 'sentence', 'treatments', 'thread_id',
                'sentiment', 'factuality', 'agrees'
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
    def prepare_treatment_definitons():
        treatment_mapping = {}
        with open("/home/johannes/talk3/data/treatment_definitons.txt", 'r') as fh:
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
            data = {"names": self.treatment_mapping[label]}

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

            data["most_popular_thread"] = group['thread_id'].value_counts().idxmax()

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
                pandas.DatetimeIndex(np.arange('2011-01', '2017-12', dtype='datetime64[M]')),
                fill_value=0
            )
            # bokeh can't read from the index, so we have to add it as a column
            month_groups['month'] = month_groups.index
            treatment_graph_data[label] = month_groups

            # break

        for rank, treatment in enumerate(sorted(
            treatment_summaries.items(), key=lambda kv: kv[1]['last_year_cnt'], reverse=True
        )):
            treatment_summaries[treatment[0]]["cnt_rank"] = rank
        return treatment_summaries, treatment_graph_data

    def data_month_crosstab(self):
        tr_mon = pandas.crosstab(self.df["treatments"], self.df["month"])
        tr_mon["sum"] = tr_mon.sum(axis=1)
        tr_mon.sort_values("sum", ascending=False, inplace=True)
        tr_mon = tr_mon.drop("sum", 1)
        head = tr_mon.head(10)
        tail = tr_mon.tail(len(tr_mon.index) - 10).sum(axis=0)
        tail.name = "Rest"
        tr_mon = head.append(tail)
        tr_mon = tr_mon.T
        return tr_mon
