import yaml
import argparse
import numpy as np
import torch


NETWORKS = ['dcnn_mc', 'drn_mc', 'unet_mc', 'drn_mcc']
NETWORK_LABEL = {'dcnn_mc': 'DN', 'drn_mc': "DRN", 'unet_mc': "U-net"}
NETWORK_WITH_LOSS = ['dcnn_mc_brier', 'drn_mc_ce', "unet_mc_ce", 'dcnn_mc_dice', 'drn_mc_dice', "unet_mc_dice"]
NETWORK_WITH_LOSS_LABEL = {'dcnn_mc_brier': "DN-Brier", 'drn_mc_ce': 'DRN-CE', "unet_mc_ce": 'U-net-CE',
                           'dcnn_mc_dice': 'DN-SD', 'drn_mc_dice': 'DRN-SD', "unet_mc_dice": 'U-net-SD'}
LOSSES = ['dice', 'ce', 'brier']


def translate_model_name(model_name):
    parts = model_name.split("_")
    if len(parts) > 2:
        model_name = parts[0] + "_" + parts[1]
    return NETWORK_LABEL[model_name]


def translate_combined_model_tag(model_tag):
    return NETWORK_WITH_LOSS_LABEL[model_tag]


def determine_plot_style(network, loss_function, mc_dropout):

    assert network in NETWORKS
    network_w_loss = network + "_" + loss_function

    marker_values = ['o', '^', 'd', '1']
    alpha_values = [0.25, 0.45, 0.45]
    # linestyles = ['-', "--", ':', '-.']
    linestyles = [(5, 2), (2, 5), (4, 10), (3, 3, 2, 2), (5, 2, 20, 2), (2, 5, 2, 20)]
    # List of Dash styles, each as integers in the format: (first line length, first space length, second line length,
    # second space length...)
    if mc_dropout:
        linecolor = 'xkcd:royal blue'
    else:
        linecolor = 'xkcd:emerald green'

    return linecolor, linestyles[NETWORK_WITH_LOSS.index(network_w_loss)], alpha_values[LOSSES.index(loss_function)], \
           marker_values[NETWORKS.index(network)]


def load_settings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp, Loader=yaml.FullLoader))
    return args


def save_settings(args, fname):
    with open(fname, 'w') as fp:
        yaml.dump(vars(args), fp)


def loadExperimentSettings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp, Loader=yaml.FullLoader))

    return args


def one_hot_encoding(labels, classes=[0, 1, 2, 3]):
    """

    :param labels: have shape [y, x] OR [z, y, x] or [2, z, y, x]  type np.int
    :return: binarized labels [4, y, x] OR [z, n_classes, y, x] OR [z, 2xn_classes, y, x] WHERE 0:4=ES and 4:8=ED
    """
    if labels.shape[0] == 2 and labels.ndim == 4:
        # shape is [2, w, h, d]
        # we are dealing with a combined ES/ED label array
        array_es = np.stack([(labels[0] == cls_idx).astype(np.int) for cls_idx in classes], axis=0)
        array_ed = np.stack([(labels[1] == cls_idx).astype(np.int) for cls_idx in classes], axis=0)
        binary_array = np.concatenate([array_es, array_ed], axis=0)
    else:
        if labels.ndim == 4:
            raise ValueError("ERROR - binarize acdc labels - shape of array is 4 but first dim != 2 (ES/ED)")
        elif labels.ndim == 2 or labels.ndim == 3:
            # shape is [x, y] OR [z, y, x]
            binary_array = np.stack([(labels == cls_idx).astype(np.int) for cls_idx in classes], axis=0)
            if labels.ndim == 3:
                binary_array = binary_array.transpose((1, 0, 2, 3))

        else:
            raise ValueError("ERROR - binarize labels acdc - Rank {} of array not supported".format(labels.ndim))

    return binary_array


def compute_entropy(pred_probs, dim=1, eps=1e-7):
    """

    :param pred_probs: shape [z, 4, x, y]
    :param eps:
    :return:
    """
    if isinstance(pred_probs, torch.Tensor):
        # convert to numpy array
        pred_probs = pred_probs.detach().cpu().numpy()
    entropy = (-pred_probs * np.log2(pred_probs + eps)).sum(axis=dim)
    entropy = np.nan_to_num(entropy)
    # we sometimes encounter tiny negative values
    umap_max = 2.
    umap_min = np.min(entropy)
    entropy = (entropy - umap_min) / (umap_max - umap_min)
    return entropy


def compute_entropy_pytorch(p, dim=1, keepdim=False, eps=1e-7):
    p = p + eps
    if keepdim:
        # return -torch.where(p > 0, p * p.log2(), p.new([0.0])).sum(dim=dim, keepdim=True)
        return -(p * p.log2()).sum(dim=dim, keepdim=True)
    else:
        # return -torch.where(p > 0, p * p.log2(), p.new([0.0])).sum(dim=dim, keepdim=True).squeeze()
        return -(p * p.log2()).sum(dim=dim, keepdim=True).squeeze()


def detect_seg_errors(labels, pred_labels, is_multi_class=False):

    """

    :param labels: if is_multi_class then [w, h] otherwise [num_of_classes, w, h]
    :param pred_labels: always [num_of_classes, w, h]
    :param is_multi_class: indicating that ground truth labels have shape [w, h]
    :return: [w, h] multiclass errors. so each voxels not equal to zero is an error. {1...nclass} indicates
                the FP-class the voxels belongs to. Possibly more than one class, but we only indicate the last one
                meaning, in the sequence of the classes
    """
    num_of_classes, w, h = pred_labels.shape
    errors = np.zeros((w, h))
    for cls in np.arange(num_of_classes):

        if is_multi_class:
            gt_labels_cls = (labels == cls).astype('int16')
        else:
            gt_labels_cls = labels[cls]
        errors_cls = gt_labels_cls != pred_labels[cls]
        errors[errors_cls] = cls

    return errors