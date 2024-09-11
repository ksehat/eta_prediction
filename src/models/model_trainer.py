from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
