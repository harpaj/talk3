import logging

from bokeh.models import Range1d, HoverTool, BoxAnnotation
from bokeh.models.sources import ColumnDataSource
from bokeh.embed import components
from bokeh.plotting import figure
import numpy as np


class GraphDrawer(object):
    def __init__(self, data_manager, config):
        self.dm = data_manager
        self.config = config
        self.treatment_score_graphs = self.get_treatment_score_graphs()
        self.treatment_count_graphs = self.get_treatment_count_graphs(relative=False)
        self.treatment_relative_graphs = self.get_treatment_count_graphs(relative=True)
        logging.info("All graph data prepared")

    @property
    def x_range(self):
        return Range1d(
            np.datetime64(self.config["range"]["start"], 'M'),
            np.datetime64(self.config["range"]["end"], 'M'),
            bounds='auto',
            min_interval=np.timedelta64(365, 'D')
        )

    def get_treatment_score_graphs(self):
        graphs = {}
        for treatment, data in self.dm.treatment_graphs.items():
            p = figure(
                y_axis_label='score',
                x_axis_type='datetime',
                x_range=self.x_range,
                tools='xwheel_zoom,xpan,reset,save',
                active_scroll='xwheel_zoom',
                active_drag='xpan',
                plot_width=800,
                plot_height=500
                # sizing_mode='stretch_both',
            )
            p.line("month", "score", line_width=2, source=data, color="black")
            pos_box = BoxAnnotation(bottom=0, top=1, fill_alpha=0.1, fill_color='#5fad56')
            neg_box = BoxAnnotation(bottom=-1, top=0, fill_alpha=0.1, fill_color='#df2935')
            p.add_layout(pos_box)
            p.add_layout(neg_box)
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

    def get_treatment_count_graphs(self, relative):
        graphs = {}
        elements = ["pos_cnt", "neu_cnt", "neg_cnt", "pos+neu_cnt", "all_cnt"]
        if relative:
            elements = ["pos_rel", "neu_rel", "neg_rel", "pos+neu_rel", "all_rel"]
        categories = elements[:3]
        for treatment, data in self.dm.treatment_graphs.items():
            areas = self.stacked(data, categories)
            months = list(data["month"])
            x2 = np.hstack((months[::-1], months))
            p = figure(
                y_axis_label='% of posts' if relative else 'posts',
                x_axis_type='datetime',
                x_range=self.x_range,
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
            p.line("month", elements[0], color="#5fad56", legend="positive", name="pos", source=data_source)
            p.line("month", elements[3], color="#f2c14e", legend="neutral", name="neu", source=data_source)
            p.line("month", elements[4], color="#df2935", legend="negative", name="neg", source=data_source)

            graphs[treatment] = components(p)
        return graphs
