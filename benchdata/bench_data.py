from abc import ABCMeta, abstractmethod


class BenchData(metaclass=ABCMeta):
    @abstractmethod
    def get_perfect_match_index(self, matrix):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def default_fillna(self):
        pass

    def default_preprocess(self):
        self.default_fillna()

