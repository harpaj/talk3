import logging

from bokeh.charts import Area
from bokeh.palettes import Inferno11
from bokeh.models import Range1d, HoverTool
from bokeh.models.sources import ColumnDataSource
from bokeh.embed import components
from bokeh.plotting import figure
import numpy as np

x_range = Range1d(np.datetime64('2011', 'Y'), np.datetime64('2018', 'Y'), bounds='auto')


class GraphDrawer(object):
    def __init__(self, data_manager):
        self.dm = data_manager
        self.summary_graph = self.get_graph_embed_data()
        self.treatment_score_graphs = self.get_treatment_score_graphs()
        self.treatment_count_graphs = self.get_treatment_count_graphs()
        logging.info("All graph data prepared")

    def get_graph_embed_data(self):
        area = Area(
            self.dm.data_month_crosstab,
            x_range=Range1d(np.datetime64('2011', 'Y'), np.datetime64('2018', 'Y'), bounds='auto'),
            tools='xwheel_zoom,xpan,reset,save',
            active_scroll='xwheel_zoom',
            active_drag='xpan',
            palette=Inferno11,
            stack=True,
            plot_width=1000
        )
        return components(area)

    def get_treatment_score_graphs(self):
        graphs = {}
        for treatment, data in self.dm.treatment_graphs.items():
            p = figure(
                y_axis_label='score',
                x_axis_type='datetime',
                x_range=x_range,
                tools='xwheel_zoom,xpan,reset,save',
                active_scroll='xwheel_zoom',
                active_drag='xpan',
                plot_width=800,
                plot_height=500
                # sizing_mode='stretch_both',
            )
            p.line("month", "score", line_width=2, source=data)
            graphs[treatment] = components(p)
        return graphs

    @staticmethod
    def stacked(df, categories):
        areas = dict()
        last = np.zeros(len(df[categories[0]]))
        for cat in categories:
            next = last + df[cat]
            areas[cat] = np.hstack((last[::-1], next))
            last = next
        return areas

    def get_treatment_count_graphs(self):
        graphs = {}
        categories = ["pos_cnt", "neu_cnt", "neg_cnt"]
        for treatment, data in self.dm.treatment_graphs.items():
            areas = self.stacked(data, categories)
            months = list(data["month"])
            x2 = np.hstack((months[::-1], months))
            p = figure(
                y_axis_label='score',
                x_axis_type='datetime',
                x_range=x_range,
                tools='xwheel_zoom,xpan,reset,save',
                active_scroll='xwheel_zoom',
                active_drag='xpan',
                plot_width=800,
                plot_height=500
                # sizing_mode='stretch_both',
            )
            hover = HoverTool(names=["pos", "neu", "neg"])
            hover.tooltips = [
                ("month", "@month{%B %Y}"),
                ("negative", "@neg_cnt"),
                ("positive", "@pos_cnt"),
                ("neutral", "@neu_cnt"),
                ("total", "@all_cnt")
            ]
            hover.formatters = {"month": "datetime"}
            p.add_tools(hover)
            p.patches(
                [x2] * len(areas), [areas[cat] for cat in categories],
                color=["#5fad56", "#f2c14e", "#df2935"], alpha=0.8, line_color=None)
            data_source = ColumnDataSource.from_df(data)
            p.line("month", "pos_cnt", color="#5fad56", legend="positive", name="pos", source=data_source)
            p.line("month", "pos+neu_cnt", color="#f2c14e", legend="neutral", name="neu", source=data_source)
            p.line("month", "all_cnt", color="#df2935", legend="negative", name="neg", source=data_source)

            graphs[treatment] = components(p)
        return graphs
