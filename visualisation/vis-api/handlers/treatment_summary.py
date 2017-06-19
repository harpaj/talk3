from .base import BaseHandler


class TreatmentSummaryHandler(BaseHandler):

    def get(self):
        treatment_list = []
        for name, summary in self.application.dm.treatment_summaries.items():
            summary["treatment"] = name
            treatment_list.append(summary)
        self.finish({"treatment_list": treatment_list})
