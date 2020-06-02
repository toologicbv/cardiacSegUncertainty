import torch
from torch.autograd import Function
import numpy as np

if torch.cuda.is_available():
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    # torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'


def get_loss_mask(predictions, ignore_labels):
    """

    :param predictions: pytorch tensor shape [batch_size, n_classes, y, x] contains softmax probabilities for classes
    :param ignore_labels: pytorch tensor of shape [batch_size, n_classes] with values {0, 1}
                          e.g. [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1]]
                               batch-size = 4 and n_classes = 3
                               The binary encoding represents which label structures to ignore in a batch slice
                               e.g. [0, 0, 1] means that we should ignore predictions in the third channel, hence
                               for class 2. E.g. for ARVC we have three classes [BG, LV, RV]. BG is never ignored
                               but often volumes only contain one target tissue structure.

    :return: torch.FloatTensor of shape [batch_size, 1, y, x]. A binary mask where 0 means that we will ignore the
                prediction for this pixel.
    """
    dflt_device = predictions.device
    # make sure we have a torch tensor object for ignore_labels (e.g. not the case during testing)
    if isinstance(ignore_labels, np.ndarray):
        ignore_labels = torch.FloatTensor(torch.from_numpy(ignore_labels).float()).to(dflt_device)
    # get indices of batch slices where we need to ignore predictions for a tissue structure that is not available
    # in the reference. ignore_label is pytorch tensor of shape [batch_size, n_classes]: the one-hot encoding of the
    # label structure we need to ignore.
    ignore_indices = torch.nonzero(ignore_labels)
    # create mask, we assume all pixels in batch contribute to loss
    batch_size, _, w, h = predictions.size()
    loss_mask = torch.ones(batch_size, 1, w, h).to(dflt_device)

    # if ignore_indices is empty then we don't need to do anything, because all slices can in principle contain all
    # tissue structures
    if len(ignore_indices) != 0:
        # get hard labels. All labels not in list of structures (e.g. [2] only RV) gets ignored
        _, res = torch.max(predictions.cpu().detach(), 1)
        # res should have shape [batch_size, y, x]
        # loop over rows (== batch_size) of ignore_indices [batch_size, 2] to set pixels in mask to zero that don't
        # contribute to loss
        for batch_id, omit_label in ignore_indices:
            loss_mask[batch_id, 0][res[batch_id] == omit_label] = 0

    return loss_mask


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        e = 1e-6
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + e

        t = (2 * self.inter.float() + e) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, (p, t) in enumerate(zip(input, target)):
        one_hot = torch.zeros(p.shape).scatter_(0, t.unsqueeze(0), 1)
        s = s + DiceCoeff().forward(p, one_hot)

    return (s / (i + 1)).squeeze()


def soft_dice_score(prob_c, one_hot):
    """

    Computing the soft-dice-loss for a SPECIFIC class according to:

    DICE_c = \frac{\sum_{i=1}^{N} (R_c(i) * A_c(i) ) }{ \sum_{i=1}^{N} (R_c(i) +   \sum_{i=1}^{N} A_c(i)  }

    Input: (1) probs: 4-dim tensor [batch_size, num_of_classes, width, height]
               contains the probabilities for each class
           (2) true_binary_labels, 4-dim tensor with the same dimensionalities as probs, but contains binary
           labels for a specific class

           Remember that classes 0-3 belongs to ES images (phase) and 4-7 to ED images

    """
    eps = 1.0e-6

    # if not isinstance(true_label_c, torch.FloatTensor) and not isinstance(true_label_c, torch.DoubleTensor):
    #     true_label_c = true_label_c.float()

    # nominator = torch.sum(one_hot * prob_c) # seems to be faster for small tensors, but slower for large tensors
    # Bob's version
    # nominator = torch.dot(one_hot.view(-1), prob_c.view(-1)) # the other way around
    # denominator = torch.sum(one_hot) + torch.sum(prob_c) + eps
    # Jorg's version: first compute loss per class and then sum all class losses
    nominator = 2 * torch.sum(one_hot * prob_c, dim=(2, 3))
    denominator = torch.sum(one_hot, dim=(2, 3)) + torch.sum(prob_c, dim=(2, 3)) + eps
    return - torch.mean(nominator/denominator)


def soft_dice_scorev2(prob_c, true_label_c):
    """

    Computing the soft-dice-loss for a SPECIFIC class according to:

    DICE_c = \frac{\sum_{i=1}^{N} (R_c(i) * A_c(i) ) }{ \sum_{i=1}^{N} (R_c(i) +   \sum_{i=1}^{N} A_c(i)  }

    Input: (1) probs: 4-dim tensor [batch_size, num_of_classes, width, height]
               contains the probabilities for each class
           (2) true_binary_labels, 4-dim tensor with the same dimensionalities as probs, but contains binary
           labels for a specific class

           Remember that classes 0-3 belongs to ES images (phase) and 4-7 to ED images

    """
    eps = 1.0e-6

    if not isinstance(true_label_c, torch.FloatTensor) and not isinstance(true_label_c, torch.DoubleTensor):
        true_label_c = true_label_c.float()
    # we sum over batches and x, y but not over classes
    nominator = torch.sum(true_label_c * prob_c, dim=(0, 2, 3))
    denominator = torch.sum(true_label_c, dim=(0, 2, 3)) + torch.sum(prob_c, dim=(0, 2, 3)) + eps
    losses = nominator/denominator
    return (-1.) * torch.sum(losses)


class DiceLossv2(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLossv2, self).__init__()
        self.n_classes = n_classes

    def forward(self, input, target):
        # if self.n_classes > 2:
        one_hot = torch.zeros(input.shape).scatter_(1, target.unsqueeze(1), 1)
        return soft_dice_scorev2(input, one_hot)


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, input, target):
        # if self.n_classes > 2:
        one_hot = torch.zeros(input.shape).scatter_(1, target.unsqueeze(1), 1)
        return soft_dice_score(input, one_hot)


class BrierLossv2(torch.nn.Module):
    def __init__(self, n_classes):
        super(BrierLossv2, self).__init__()
        self.n_classes = n_classes

    def forward(self, softmax_probs, target_labels):
        """

        :param softmax_probs: with shape [batch, n_classes, x, y]
        :param target_labels: [batch, x, y] multi-class labels (NOT ONE-HOT)
        :return:
        """
        # convert ref labels to one-hot encoding
        one_hot_target = torch.zeros(softmax_probs.shape).scatter_(1, target_labels.unsqueeze(1), 1)

        # only average over batch, original Brier averages over classes (dim=1)
        denominator = target_labels.size(0)
        return 1 / denominator * torch.sum((one_hot_target - softmax_probs) ** 2)


class BrierLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(BrierLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, softmax_probs, target_labels):
        """

        :param softmax_probs: with shape [batch, n_classes, x, y]
        :param target_labels: [batch, x, y] multi-class labels (NOT ONE-HOT)
        :return:
        """
        # convert ref labels to one-hot encoding
        one_hot_target = torch.zeros(softmax_probs.shape).scatter_(1, target_labels.unsqueeze(1), 1)

        # only average over batch, original Brier averages over classes (dim=1)
        denominator = target_labels.size(0)
        # return 1 / denominator * torch.sum(torch.mean((one_hot_target - softmax_probs) ** 2, dim=1))
        return torch.mean((one_hot_target - softmax_probs) ** 2)


def soft_dice_loss(prob_c, true_label_c):
    """

    Computing the soft-dice-loss for a SPECIFIC class according to:

    DICE_c = \frac{\sum_{i=1}^{N} (R_c(i) * A_c(i) ) }{ \sum_{i=1}^{N} (R_c(i) +   \sum_{i=1}^{N} A_c(i)  }

    Input: (1) probs: 4-dim tensor [batch_size, num_of_classes, width, height]
               contains the probabilities for each class
           (2) true_binary_labels, [z, y, x]

           Remember that classes 0-3 belongs to ES images (phase) and 4-7 to ED images

    """
    eps = 1.0e-6
    n_classes = prob_c.size(1)
    true_label_c = torch.zeros(prob_c.shape).scatter_(1, true_label_c.unsqueeze(1), 1)
    if not isinstance(true_label_c, torch.FloatTensor) and not isinstance(true_label_c, torch.DoubleTensor):
        true_label_c = true_label_c.float()
    losses = torch.FloatTensor(n_classes).cuda()
    for cls_idx in np.arange(n_classes):
        losses[cls_idx] = torch.sum(true_label_c[:, cls_idx] * prob_c[:, cls_idx]) / \
                          (torch.sum(true_label_c[:, cls_idx]) + torch.sum(prob_c[:, cls_idx]) + eps)

    return (-1.) * torch.sum(losses)


def brier_score_loss(prob_c, true_label_c):
    """

    :param prob_c: torch tensor [batch_size, nclasses, x, y]
    :param true_label_c: torch tensor [batch_size, x, y] <--- NOT one-hot encoded reference label
    :return: torch scalar
    """
    true_label_c = torch.zeros(prob_c.size()).scatter_(1, true_label_c.unsqueeze(1), 1)
    true_label_c = true_label_c.to(device)
    if not isinstance(true_label_c, torch.FloatTensor) and not isinstance(true_label_c, torch.DoubleTensor):
        true_label_c = true_label_c.float()
    # PERFORMANCE IS REALLY BAD we average over: batch_size * x * y (so omitting #classes)
    # only on batch is better (but higher HD than my previous solution, sum x, y average over classes and batch
    denominator = true_label_c.size(0) # * true_labels.size(2) * true_labels.size(3)
    loss = 1/denominator * torch.sum((true_label_c - prob_c)**2)

    return loss
