import pandas
from tornado.options import options


class DataManager(object):
    def __init__(self):
        self.df = self.prepare_initial_dataframe()

    @staticmethod
    def prepare_initial_dataframe():
        df = pandas.read_csv(
            open("/home/johannes/talk3/data/treatment_detected.csv", 'r'),
            usecols=['subforum', 'post_id', 'timestamp', 'sentence', 'treatments'],
            index_col=None,
            parse_dates=['timestamp'],
            infer_datetime_format=True
        )
        df['month'] = df['timestamp'].values.astype('<M8[M]')
        return df

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
