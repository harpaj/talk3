import csv
import logging
from datetime import datetime

import pandas
import numpy as np


class DataManager(object):
    def __init__(self):
        self.df = self.prepare_initial_dataframe()
        self.data_month_crosstab = self.data_month_crosstab()
        self.treatment_mapping = self.prepare_treatment_definitons()
        self.treatment_summaries = self.prepare_treatment_summaries()
        logging.info("All data calculated")

    @staticmethod
    def prepare_initial_dataframe():
        df = pandas.read_csv(
            open("/home/johannes/talk3/data/treatment_detected_linewise.csv", 'r'),
            usecols=[
                'subforum', 'post_id', 'timestamp', 'sentence', 'treatments', 'thread_id'
            ],
            index_col=None,
            parse_dates=['timestamp'],
            infer_datetime_format=True
        )
        df['month'] = df['timestamp'].values.astype('<M8[M]')
        return df

    @staticmethod
    def prepare_treatment_definitons():
        treatment_mapping = {}
        with open("/home/johannes/talk3/data/treatment_definitons.txt", 'r') as fh:
            reader = csv.reader(fh)
            for line in reader:
                treatment_mapping[line[0]] = line[1:]
        return treatment_mapping

    def prepare_treatment_summaries(self):
        treatment_summaries = {}
        one_year_ago = np.datetime64(datetime.now(), 'D') - np.timedelta64(365, 'D')
        two_years_ago = one_year_ago - np.timedelta64(365, 'D')
        last_year_total = len(self.df[self.df["timestamp"] > one_year_ago])
        previous_year_total = len(self.df[
            (self.df["timestamp"] > two_years_ago) & (self.df["timestamp"] < one_year_ago)])
        for label, group in self.df.groupby("treatments"):
            data = {"names": self.treatment_mapping[label]}
            data["last_year_cnt"] = len(group[
                group["timestamp"] > one_year_ago
            ])
            data["previous_year_cnt"] = len(group[
                (group["timestamp"] > two_years_ago) & (group["timestamp"] < one_year_ago)
            ])
            data["most_popular_thread"] = group['thread_id'].value_counts().idxmax()
            data["last_year_%"] = data["last_year_cnt"] / last_year_total
            data["previous_year_%"] = data["previous_year_cnt"] / previous_year_total
            treatment_summaries[label] = data
        for rank, treatment in enumerate(sorted(
            treatment_summaries.items(), key=lambda kv: kv[1]['last_year_cnt'], reverse=True
        )):
            treatment_summaries[treatment[0]]["cnt_rank"] = rank
        return treatment_summaries

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
