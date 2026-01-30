
class ActionOptimizationModule:

    def __init__(self, settings: dict):
        self.actions = settings.actions

    def get_optimal_action_set(self, state):
        # Calcolare prima ActionSetFeaseble tramite regole hard coded, per esempio se ci si trova in
        # uno stato non si può fare un'azione, per ora implementiamo la Rag 
        pass