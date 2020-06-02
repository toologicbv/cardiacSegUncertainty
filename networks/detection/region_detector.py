import torch.nn as nn
import math

import torch
from networks.detection.vgg_style_model import make_layers


class RegionDetector(nn.Module):

    def __init__(self, n_classes=2, n_channels=3, use_batch_norm=True, drop_prob=0.5):
        super(RegionDetector, self).__init__()
        self.nclasses = n_classes
        self.num_input_channels = n_channels
        self.use_batch_norm = use_batch_norm
        self.drop_prob = drop_prob

        # The base VGG net. 3 convolutional layers (F=3x3; p=1) with BatchNorm + ReLU + MaxPooling (2x2).
        # So we have an output_stride (compression) factor of 8. e.g. 80x80 patch size results in 10x10 activation map
        self.base_model = make_layers(num_of_input_channels=self.num_input_channels, cfg=[16, 'M', 16, 32, 'M', 32, 'M', 64],
                                      batch_norm=self.use_batch_norm)
        self.softmax_layer = nn.Softmax(dim=1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)

        self.classifier = None
        self.classifier_extra = None
        self._make_classifier()
        self._initialize_weights()
        self._compute_num_trainable_params()

    def _make_classifier(self):
        # get the last sequential module,
        # we have 2 options, depending on whether or not we use BatchNormalization:
        # (1) [conv2d, nn.ReLU(inplace=True)]   (2) [Conv2d, Batchnorm, ReLU]
        #     we're looking for the parameter specifying the number of output channels in the conv2d layer
        #     Hence, for option (1) we choose list[-2] and for (2) list[-3] as target layer that contains the params
        #     Conv2d has e.g. params [(32, 16, 2, 2), (16,)]). So below, we extract [0][1] = 16 in this case

        if self.use_batch_norm:
            last_base_layer = self.base_model[-3]
        else:
            last_base_layer = self.base_model[-2]

        # param.shape in comprehension should have shape (#output_chnls, #inputchnls, kernel_size, kernel_size)
        # e.g. for Conv2d with ic=16 and oc=32 we have: params [(32, 16, 2, 2), (32,)])
        num_of_channels_last_layer = [param.shape for param in last_base_layer.parameters()][1][0]
        print(("INFO - RegionDetector - debug - num_of_channels_last_layer {}".format(num_of_channels_last_layer)))

        self.classifier = nn.Sequential(
            nn.Conv2d(num_of_channels_last_layer, 128, kernel_size=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(128, 128, kernel_size=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(p=self.drop_prob),
            nn.Conv2d(128, self.nclasses, kernel_size=1, padding=0),
        )

    def forward(self, x):

        x = self.base_model(x)
        x = self.classifier(x)
        # we return softmax probs and log-softmax. The last for computation of NLLLoss
        out = {"softmax": self.softmax_layer(x),
               "log_softmax": self.log_softmax_layer(x)}
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _compute_num_trainable_params(self):
        self.model_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def sum_grads(self, verbose=False):
        sum_grads = 0.
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_grads += torch.sum(torch.abs(param.grad.data))
            else:
                if verbose:
                    print(("WARNING - No gradients for parameter >>> {} <<<".format(name)))

        return sum_grads


