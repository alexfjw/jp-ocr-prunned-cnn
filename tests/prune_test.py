from src.prune import *
from src.nn.models import ChineseNet
import unittest

class TestPruneProcess(unittest.TestCase):

    def setUp(self):
        self.model = ChineseNet(3156)

    def estimatePruningIterations_ShouldGetRightNum(self):
        estimate_pruning_iterations()
        pass


    def getNumFeatureMaps_ShouldGetRightNum(self):
        pass


    def getNumParameters_ShouldGetRightNum(self):
        pass


    def pruneStep_ShouldPruneModule(self):
        pass


if __name__ == '__main__':
    unittest.main()