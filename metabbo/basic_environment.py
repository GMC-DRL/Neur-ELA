class PBO_Env:
    """
    An environment with a problem and an optimizer.
    """
    def __init__(self,
                 problem,
                 optimizer,
                 ):
        self.problem = problem
        self.optimizer = optimizer

    def reset(self):
        return self.optimizer.init_population(self.problem)

    def step(self, action):
        return self.optimizer.update(action, self.problem)