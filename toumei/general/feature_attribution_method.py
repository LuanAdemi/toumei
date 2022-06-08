from toumei.general.objective import Objective
from toumei.models.explanation_models import LinearExplanationModel


class FeatureAttributionMethod(Objective):
    def __init__(self):
        super(FeatureAttributionMethod, self).__init__()
        self.explanation_model = LinearExplanationModel()


