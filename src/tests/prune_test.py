from src.prune import *
from src.nn.models import ChineseNet
from src.nn.prunable_nn import PConv2d
import torch.nn as nn
import unittest

class TestPruneProcess(unittest.TestCase):

    def setUp(self):
        self.model = ChineseNet(3156)

    def test_getNumParameters_ShouldGetRightNum(self):
        pconv2d = PConv2d(3,3,3)
        # out channels + num_filters*filter_size^2*depth
        num_params = 3 + 3*3*3*3

        self.assertEqual(get_num_parameters(pconv2d), num_params)

    def test_getNumPrunableFeatureMaps_ShouldGetRightNum(self):
        self.assertEqual(get_num_prunable_feature_maps(self.model), 1664)



if __name__ == '__main__':
    unittest.main()