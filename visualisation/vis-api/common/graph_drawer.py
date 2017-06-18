from bokeh.charts import Area
from bokeh.palettes import Inferno11
from bokeh.models import Range1d
from bokeh.embed import components
import numpy as np


class GraphDrawer(object):
    def __init__(self, data_manager):
        self.dm = data_manager

    def get_graph_embed_data(self):
        area = Area(
            self.dm.data_month_crosstab(),
            x_range=Range1d(np.datetime64('2011', 'Y'), np.datetime64('2018', 'Y'), bounds='auto'),
            tools='xwheel_zoom,xpan,reset,save',
            active_scroll='xwheel_zoom',
            active_drag='xpan',
            palette=Inferno11,
            stack=True,
            plot_width=1000
        )
        return components(area)
