from .base import BaseHandler


class TreatmentSummaryHandler(BaseHandler):

    def get(self):
        treatment_list = list(self.application.dm.treatment_summaries.values())
        self.finish({"treatment_list": treatment_list})
