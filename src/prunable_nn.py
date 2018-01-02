import torch.nn as nn
import torch


class PConv2d(nn.Conv2d):
    """
    Exactly like a Conv2d, but saves the activation of the last forward pass
    This allows calculation of the taylor estimate in https://arxiv.org/abs/1611.06440
    Includes convenience functions for feature map pruning
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.__recent_activations = None
        self.taylor_estimates = None
        self.register_backward_hook(self.__estimate_taylor_importance)

    def forward(self, x):
        output = super().forward(x)
        self.__recent_activations = output.clone()
        return output

    def __estimate_taylor_importance(self, grad_input, grad_output):
        # skip dim=1, its the dim for feature maps
        n_batch, _, n_x, n_y = self.__recent_activations.size()
        n_dimensions = n_batch * n_x * n_y
        estimates = self.__recent_activations.mul_(grad_output) \
            .sum(dim=3) \
            .sum(dim=2) \
            .sum(dim=0) \
            .div_(n_dimensions)

        # normalization
        self.taylor_estimates = estimates / torch.sqrt(torch.sum(estimates * estimates))
        del estimates, self.__recent_activations
        self.__recent_activations = None

    def prune_feature_map(self, map_index):
        self.weight = self.weight.cuda()

        num_feature_maps, *others = self.weight.size()
        temp = nn.Parameter(torch.Tensor(num_feature_maps - 1, *others)).cuda()
        temp[:map_index, :, :, :] = self.weight[:map_index, :, :, :]
        temp[map_index:, :, :, :] = self.weight[map_index+1:, :, :, :]
        del self.weight
        self.weight = temp
        self.out_channels -= 1

    def drop_input_channel(self, index):
        """
        Use when a convnet earlier in the chain is prunned. Reduces input channel count
        :param index:
        :return:
        """
        self.weight = self.weight.cuda()

        num_feature_maps, channels, *kernel_size = self.weight.size()
        temp = nn.Parameter(torch.Tensor(num_feature_maps, channels-1, *kernel_size)).cuda()
        temp[:, :index, :, :] = self.weight[:, :index, :, :]
        temp[:, index:, :, :] = self.weight[:, index+1:, :, :]
        del self.weight
        self.weight = temp
        self.in_channels -= 1


class PBatchNorm2d(nn.BatchNorm2d):

    def drop_input_channel(self, index):
        self.num_features -= 1

        if self.affine:
            self.weight = self.weight.cuda()
            new_weight = nn.Parameter(torch.Tensor(self.num_features)).cuda()
            new_bias = nn.Parameter(torch.Tensor(self.num_features)).cuda()

            new_weight[:index] = self.weight[:index]
            new_weight[index:] = self.weight[index+1:]
            new_bias[:index] = self.weight[:index]
            new_bias[index:] = self.weight[index+1:]

            del self.weight, self.bias
            self.weight = new_weight
            self.bias = new_bias



