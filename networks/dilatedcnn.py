import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Independent, Normal

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'

DEFAULT_DCNN_2D = {'kernels': [3, 3, 3, 3, 3, 3, 3, 3, 1],
                   'channels': [32, 32, 32, 32, 32, 32, 64, 128, 128],
                   'dilation': [1, 1, 2, 4, 8, 16, 32, 1, 1],
                   'stride': [1, 1, 1, 1, 1, 1, 1, 1, 1],
                   'batch_norm': [False, True, True, True, True, True, True, True, True],
                   'non_linearity': [nn.ELU, nn.ELU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ELU, nn.ELU],
                   'dropout': [0., 0., 0., 0., 0., 0., 0., 0.5, 0.5]
                   }


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


class DilatedCNN2D(nn.Module):

    def __init__(self, n_input=1, n_classes=4):
        # receptive field [131, 131]
        super().__init__()
        channels = DEFAULT_DCNN_2D['channels']
        kernels = DEFAULT_DCNN_2D['kernels']
        dilations = DEFAULT_DCNN_2D['dilation']
        non_linearities = DEFAULT_DCNN_2D['non_linearity']
        batch_norms = DEFAULT_DCNN_2D['batch_norm']
        dropouts = DEFAULT_DCNN_2D['dropout']
        self.net = nn.Sequential()
        for layer_id, num_channels in enumerate(channels):
            layer = self._make_conv_layer(layer_id, n_input, num_channels, kernels[layer_id], dilations[layer_id],
                                          dropouts[layer_id], batch_norms[layer_id], non_linearities[layer_id])
            self.net = nn.Sequential(self.net, layer)
            n_input = num_channels

        self.net.add_module('last_layer', nn.Conv2d(128, n_classes, 1, dilation=1))
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.apply(weights_init)

    def _make_conv_layer(self, layer_id, channels_in, channels_out, kernel=3, dilation=1, p_dropout=0.,
                         batch_norm=False, non_linearity=nn.ReLU):
        layer = nn.Sequential()
        layer.add_module("conv2d_" + str(layer_id), nn.Conv2d(channels_in, channels_out, kernel, dilation=dilation))
        layer.add_module("non_linear_" + str(layer_id), non_linearity())
        if batch_norm:
            layer.add_module("batchnorm_" + str(layer_id), nn.BatchNorm2d(channels_out))
        if p_dropout != 0:
            layer.add_module("dropout_layer" + str(layer_id), nn.Dropout2d(p=p_dropout))
        return layer

    def forward(self, x):
        out = self.net(x)
        return {'log_softmax': self.log_softmax(out), 'softmax': self.softmax(out)}


class BayesDilatedCNN2D(DilatedCNN2D):
    dropout_layer = "dropout_layer"

    def __init__(self, n_input=1, n_classes=4):
        # receptive field [131, 131]
        super().__init__()
        channels = DEFAULT_DCNN_2D['channels']
        kernels = DEFAULT_DCNN_2D['kernels']
        dilations = DEFAULT_DCNN_2D['dilation']
        non_linearities = DEFAULT_DCNN_2D['non_linearity']
        batch_norms = DEFAULT_DCNN_2D['batch_norm']
        dropouts = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.net = nn.Sequential()
        for layer_id, num_channels in enumerate(channels):
            layer = self._make_conv_layer(layer_id, n_input, num_channels, kernels[layer_id], dilations[layer_id],
                                          dropouts[layer_id], batch_norms[layer_id], non_linearities[layer_id])
            self.net = nn.Sequential(self.net, layer)
            n_input = num_channels

        self.net.add_module('last_layer', nn.Conv2d(128, n_classes, 1, dilation=1))
        self.apply(weights_init)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def train(self, mode=True, mc_dropout=False):
        """ Sets the module in training mode.
            !!! OVERWRITING STANDARD PYTORCH METHOD for nn.Module

            OR
                if mc_dropout=True and mode=False (use dropout during inference) we set all modules
                to train-mode=False except for DROPOUT layers
                In this case it is important that the module_name matches BayesDRNSeg.dropout_layer

        Returns:
            Module: self
        """
        self.training = mode
        for module_name, module in self.named_modules():
            module.training = mode
            if mc_dropout and not mode:
                if BayesDilatedCNN2D.dropout_layer in module_name:
                    # print("nn.Module.train - {}".format(module_name))
                    module.training = True

        return self

    def eval(self, mc_dropout=False):
        """Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.
        """
        return self.train(False, mc_dropout=mc_dropout)


class CombinedBayesDilatedCNN2D(BayesDilatedCNN2D):

    def __init__(self, n_input=1, n_classes=4, sigma=1., samples=5):
        super().__init__()
        self.sigma = sigma
        self.samples = samples
        channels = DEFAULT_DCNN_2D['channels']
        kernels = DEFAULT_DCNN_2D['kernels']
        dilations = DEFAULT_DCNN_2D['dilation']
        non_linearities = DEFAULT_DCNN_2D['non_linearity']
        batch_norms = DEFAULT_DCNN_2D['batch_norm']
        dropouts = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.net = nn.Sequential()
        for layer_id, num_channels in enumerate(channels):
            layer = self._make_conv_layer(layer_id, n_input, num_channels, kernels[layer_id], dilations[layer_id],
                                          dropouts[layer_id], batch_norms[layer_id], non_linearities[layer_id])
            self.net = nn.Sequential(self.net, layer)
            n_input = num_channels

        self.segmentation = nn.Conv2d(128, n_classes, 1, dilation=1)
        self.aleatoric_unc = nn.Conv2d(128, 1, 1, dilation=1)
        self.apply(weights_init)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.net(x)
        y = self.segmentation(features)
        max_logits, _ = torch.max(y, dim=1)
        scale = torch.exp(- self.aleatoric_unc(features))

        return {'log_softmax': self.log_softmax(y), 'softmax': self.softmax(y),
                'sigma': scale, 'logits': y}


if __name__ == "__main__":
    import torchsummary

    model = BayesDilatedCNN2D(n_input=1, n_classes=4)
    model = model.to('cuda')
    torchsummary.summary(model, (1, 256, 256))
