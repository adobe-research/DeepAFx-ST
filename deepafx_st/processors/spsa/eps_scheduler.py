import torch


class EpsilonScheduler:
    def __init__(
        self,
        epsilon: float = 0.001,
        patience: int = 10,
        factor: float = 0.5,
        verbose: bool = False,
    ):
        self.epsilon = epsilon
        self.patience = patience
        self.factor = factor
        self.best = 1e16
        self.count = 0
        self.verbose = verbose

    def step(self, metric: float):

        if metric < self.best:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.verbose:
                print(f"Train loss has not improved for {self.count} epochs.")
            if self.count >= self.patience:
                self.count = 0
                self.epsilon *= self.factor
                if self.verbose:
                    print(f"Reducing epsilon to {self.epsilon:0.2e}...")
