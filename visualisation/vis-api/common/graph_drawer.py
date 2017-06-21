import logging

from bokeh.charts import Area
from bokeh.palettes import Inferno11
from bokeh.models import Range1d
from bokeh.embed import components
from bokeh.plotting import figure
import numpy as np


class GraphDrawer(object):
    def __init__(self, data_manager):
        self.dm = data_manager
        self.summary_graph = self.get_graph_embed_data()
        self.treatment_sentiment_graphs = self.get_treament_sentiment_graphs()
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

    def get_treament_sentiment_graphs(self):
        graphs = {}
        x_range = Range1d(np.datetime64('2011', 'Y'), np.datetime64('2018', 'Y'), bounds='auto')
        for treatment, data in self.dm.treatment_graphs.items():
            p = figure(
                y_axis_label='score',
                x_axis_type='datetime',
                x_range=x_range,
                tools='xwheel_zoom,xpan,reset,save',
                active_scroll='xwheel_zoom',
                active_drag='xpan',
                # sizing_mode='stretch_both',
            )
            p.line("month", "score", line_width=2, source=data)
            graphs[treatment] = components(p)
        return graphs
