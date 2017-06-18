import tornado.ioloop
import tornado.web
import tornado.options

from common.data_manager import DataManager
from common.graph_drawer import GraphDrawer


class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def get(self):
        script, div = self.application.gd.get_graph_embed_data()
        self.finish({
            "script": script[32:-9],
            "div": div
        })

    def options(self):
        self.set_status(204)
        self.finish()


class VisApplication(tornado.web.Application):
    def __init__(self, handlers, **settings):
        self.dm = DataManager()
        self.gd = GraphDrawer(self.dm)
        super(VisApplication, self).__init__(handlers, **settings)


def make_app():
    return VisApplication([
        (r"/", MainHandler)
    ])


if __name__ == "__main__":
    tornado.options.parse_config_file("config.cfg")
    app = make_app()
    app.listen(8765)
    tornado.ioloop.IOLoop.current().start()
