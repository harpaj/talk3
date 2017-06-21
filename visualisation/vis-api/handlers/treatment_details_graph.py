from .base import BaseHandler


class TreatmentDetailsGraphHandler(BaseHandler):

    def get(self, treatment):
        script, div = self.application.gd.treatment_sentiment_graphs[treatment]
        self.finish({
            "script": script[32:-9],
            "div": div
        })
