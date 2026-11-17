from abc import ABC, abstractmethod


class Experiment(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def setup(self):
        """Initialize model, optimizer, scheduler, dataloader, wandb, etc."""
        pass

    @abstractmethod
    def train(self):
        """Training loop with logger tracking."""
        pass

    @abstractmethod
    def validate(self):
        """Evaluation on validation set. Logs metrics."""
        pass

    @abstractmethod
    def test(self):
        """Evaluation on test set. Logs metrics."""
        pass

    @abstractmethod
    def run(self):
        """Full run: setup -> train -> evaluate"""
        pass