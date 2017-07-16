import logging
import configparser

import tornado.ioloop
import tornado.web
import tornado.options
import tornado.log

from common.data_manager import DataManager
from common.graph_drawer import GraphDrawer
from handlers.treatment_details_graph import TreatmentDetailsGraphsHandler
from handlers.treatment_details import TreatmentDetailsHandler
from handlers.treatment_summary import TreatmentSummaryHandler


class VisApplication(tornado.web.Application):
    def __init__(self, handlers, **settings):
        config = read_config()
        self.dm = DataManager(config)
        self.gd = GraphDrawer(self.dm, config)
        super(VisApplication, self).__init__(handlers, **settings)


def read_config():
    config = configparser.ConfigParser()
    config.read("config.cfg")
    return config


def make_app():
    return VisApplication([
        (r"/graphs/([^\/]+)/(\w+)", TreatmentDetailsGraphsHandler),
        (r"/treatment/(.+)", TreatmentDetailsHandler),
        (r"/treatment_summary", TreatmentSummaryHandler)
    ])


if __name__ == "__main__":
    tornado.log.enable_pretty_logging()
    app = make_app()
    app.listen(8765)
    logging.info("API listening on port 8765")
    tornado.ioloop.IOLoop.current().set_blocking_log_threshold(0.05)
    tornado.ioloop.IOLoop.current().start()
