import torch.nn as nn
import torch


class Input3DSlab(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Input3DSlab, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=(0, 1, 1))
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # input x has shape [batch_dim, in_channels, depth, w, h]
        x = self.conv3d(x)
        # because we used no padding for the depth dimension and our input always has size 3 for the depth dim
        # we end up with a shape of [batch_dim, out_channels, 1, w, h] and hence we squeeze the depth dimension
        # which is dim=2
        x = torch.squeeze(x, dim=2)
        return self.relu(self.batch_norm(x))


def make_layers(num_of_input_channels, cfg, batch_norm=False):
    """

    :param num_of_input_channels:
    :param cfg: list of parameters:
        Indices:
        0 = number of input channels

    :param batch_norm:

    :return:
    """
    layers = []
    in_channels = num_of_input_channels
    # 1st position specifies #input channels (to start with) hence we omit that index in the loop
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]
        else:
            conv_layer = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv_layer, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv_layer, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


