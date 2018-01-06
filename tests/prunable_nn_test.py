import unittest
import src.prunable_nn as pnn
import torch
import numpy as np
from torch.autograd import Variable


class TestPrunableConv2d(unittest.TestCase):

    def setUp(self):
        torch.cuda.manual_seed_all(1)
        torch.manual_seed(1)

        self.module = pnn.PConv2d(5, 5, 3, padding=1).cuda()
        self.module.train()

        self.input_shape = (1, 5, 50, 50)
        self.input = Variable(torch.rand(*self.input_shape).cuda(), requires_grad=True)
        self.upstream_gradient = torch.rand(*self.input_shape).cuda()

    def test_taylor_estimates(self):
        output = self.module(self.input)
        torch.autograd.backward(output, self.upstream_gradient)

        # ensure input and output are different
        self.assertFalse(np.array_equal(self.input.data.cpu().numpy(), output.data.cpu().numpy()))

        estimates = self.module.taylor_estimates.data.cpu()
        size = estimates.size()

        # ensure sane size
        self.assertEqual(size, torch.FloatTensor(self.input_shape[1]).size())
        # ensure not zero
        self.assertFalse(np.array_equal(estimates.numpy(), torch.zeros(size).numpy()))

    def test_prune_feature_map(self):
        dropped_index = 0
        output = self.module(self.input)
        torch.autograd.backward(output, self.upstream_gradient)

        old_weight_size = self.module.weight.size()
        old_bias_size = self.module.bias.size()
        old_out_channels = self.module.out_channels
        old_weight_values = self.module.weight.data.cpu().numpy()

        # ensure that the chosen index is dropped
        self.module.prune_feature_map(dropped_index)

        # check bias size
        self.assertEqual(self.module.bias.size()[0], (old_bias_size[0]-1))
        # check output channels
        self.assertEqual(self.module.out_channels, old_out_channels-1)

        _, *other_old_weight_sizes = old_weight_size
        # check weight size
        self.assertEqual(self.module.weight.size(), (old_weight_size[0]-1, *other_old_weight_sizes))
        # check weight value
        expected = np.delete(old_weight_values, dropped_index , 0)
        self.assertTrue(np.array_equal(self.module.weight.data.cpu().numpy(), expected))

    def test_drop_input_channel(self):
        dropped_index = 0

        old_weight_size = self.module.weight.size()
        old_in_channels = self.module.in_channels
        old_weight_values = self.module.weight.data.cpu().numpy()

        # ensure that the chosen index is dropped
        self.module.drop_input_channel(dropped_index)
        expected = np.delete(old_weight_values, dropped_index , 1)
        self.assertTrue(np.array_equal(self.module.weight.data.cpu().numpy(), expected))


class TestDropInputClasses(unittest.TestCase):

    def test_PLinear(self):
        pass

    def test_PBatchNorm2d(self):
        pass


if __name__ == '__main__':
    unittest.main()