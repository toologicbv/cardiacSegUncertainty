import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CycleLR(_LRScheduler):
    """

        Implementation of cyclic learning rate adjustment in order to generate snapshot ensembles as described
        in the paper
        Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J.E., Weinberger, K.Q.: Snapshot
        ensembles: Train 1, get m for free. arXiv preprint arXiv:1704.00109 (2017)

    """
    def __init__(self, optimizer, alpha_zero=0.2, cycle_length=10000, last_epoch=-1):

        self.alpha_zero = alpha_zero
        self.cycle_length = cycle_length
        self.lr = alpha_zero
        super(CycleLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch

        self.lr = (self.alpha_zero / 2.0) * \
             (np.cos((np.pi * float(np.mod(epoch, self.cycle_length))) / self.cycle_length) + 1)
        return [self.lr for _ in self.base_lrs]
