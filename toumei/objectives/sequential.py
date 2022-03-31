from objective import Objective


class Sequential(Objective):
    def __init__(self, *objectives):
        super(Sequential, self).__init__()
        