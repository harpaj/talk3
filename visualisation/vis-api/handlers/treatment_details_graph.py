from .base import BaseHandler


class TreatmentDetailsGraphsHandler(BaseHandler):

    def get(self, treatment, graph):
        if graph == "score":
            script, div = self.application.gd.treatment_score_graphs[treatment]
        elif graph == "count":
            script, div = self.application.gd.treatment_count_graphs[treatment]
        elif graph == "relative":
            script, div = self.application.gd.treatment_relative_graphs[treatment]
        self.finish({
            "script": script[32:-9],
            "div": div
        })
