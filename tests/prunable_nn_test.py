import unittest
import src.prunable_nn as pnn
import torch
from torch.autograd import Variable


class TestPrunableConv2d(unittest.TestCase):

    def setUp(self):
        torch.cuda.manual_seed_all(1)
        torch.manual_seed(1)

        self.module = pnn.PConv2d(5, 5, 3, padding=1).cuda()
        self.module.train()

        shape = (1, 5, 50, 50)
        self.input = Variable(torch.rand(*shape).cuda(), requires_grad=True)
        self.upstream_gradient = torch.rand(*shape).cuda()

    def test_taylor_estimates(self):
        output = self.module(self.input)
        torch.autograd.backward(output, self.upstream_gradient)

        # todo: convert to assert, should not be zero
        print(self.input.data.cpu())
        print(output.data.cpu())

        # todo: ensure 1d, not zero & of sane size
        print(self.module.taylor_estimates.data.cpu())


    def test_prune_feature_map(self):
        pass

    def test_drop_input_channel(self):
        pass


class TestDropInputClasses(unittest.TestCase):

    def test_PLinear(self):
        pass

    def test_PBatchNorm2d(self):
        pass


if __name__ == '__main__':
    unittest.main()