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
        pass

    def drop_input_channel(self, index):
        """
        Used when a previous conv net was prunned, reducing input channel count
        :param index:
        :return:
        """
        pass


class PBatchNorm2d(nn.BatchNorm2d):
    # TODO
    pass



