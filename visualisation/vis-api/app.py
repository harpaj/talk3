import logging

import tornado.ioloop
import tornado.web
import tornado.options

from common.data_manager import DataManager
from common.graph_drawer import GraphDrawer
from handlers.treatment_frequency_graph import TreatmentFrequencyGraphHandler
from handlers.treatment_details_graph import TreatmentDetailsGraphsHandler
from handlers.treatment_details import TreatmentDetailsHandler
from handlers.treatment_summary import TreatmentSummaryHandler


class VisApplication(tornado.web.Application):
    def __init__(self, handlers, **settings):
        self.dm = DataManager()
        self.gd = GraphDrawer(self.dm)
        super(VisApplication, self).__init__(handlers, **settings)


def make_app():
    return VisApplication([
        (r"/graphs/treatment_frequency", TreatmentFrequencyGraphHandler),
        (r"/graphs/([^\/]+)/(\w+)", TreatmentDetailsGraphsHandler),
        (r"/treatment/(.+)", TreatmentDetailsHandler),
        (r"/treatment_summary", TreatmentSummaryHandler)
    ])


if __name__ == "__main__":
    tornado.options.parse_config_file("config.cfg")
    app = make_app()
    app.listen(8765)
    logging.info("API listening on port 8765")
    tornado.ioloop.IOLoop.current().set_blocking_log_threshold(0.05)
    tornado.ioloop.IOLoop.current().start()
