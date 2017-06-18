from .base import BaseHandler


class TreatmentSummaryHandler(BaseHandler):

    def get(self):
        summary = self.application.dm.treatment_summaries
        self.finish(summary)
