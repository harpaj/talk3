from .base import BaseHandler


class TreatmentFrequencyGraphHandler(BaseHandler):

    def get(self):
        script, div = self.application.gd.summary_graph
        self.finish({
            "script": script[32:-9],
            "div": div
        })
