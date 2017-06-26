from .base import BaseHandler


class TreatmentDetailsHandler(BaseHandler):

    def get(self, treatment):
        data = self.application.dm.treatment_detailed_data[treatment]
        self.finish(data)
