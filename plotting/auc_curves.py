import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
# from sklearn.utils.fixes import signature
from inspect import signature
from sklearn.metrics import auc
import copy

from networks.detection.general_setup import config_detector


def plot_detection_auc_curves(slice_precision, slice_sensitivity, slice_fp, slice_fp_rate, detection_rate, region_fp,
                              width=8, height=8, do_show=True, do_save=False, output_dir=None, model_tag=None):

    step_kwargs = ({'step': 'pre'} if 'step' in signature(plt.fill_between).parameters else {})
    recall = [0] + copy.deepcopy(slice_sensitivity)
    precision = [1] + copy.deepcopy(slice_precision)
    rec_diff = np.array(recall[1:]) - np.array(recall[:-1])
    average_precision = np.sum(np.array(precision[1:]) * rec_diff)
    # recall on x-axis and precision on y-axis.
    # x_recall = np.linspace(0, 1, 50)
    # y_precision = np.interp(x_recall, np.array(recall), np.array(precision))
    # recall = copy.deepcopy(slice_sensitivity)
    # precision = copy.deepcopy(slice_precision)
    _ = plt.figure(figsize=(width, height))
    plt.step(recall, precision, color='b', alpha=0.2, where=step_kwargs['step'], label=model_tag)
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    # plt.plot(slice_sensitivity, slice_precision, alpha=0.2, color='b', linestyle='-')
    plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
    plt.xlabel('Recall', **config_detector.axis_font18)
    plt.ylabel('Precision', **config_detector.axis_font18)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    if model_tag is not None:
        plt.legend(loc="best", prop={'size': 18})
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title('Precision/Recall-slices: AP={:.2f}'.format(average_precision),
              **config_detector.axis_font18)  # Mean per patient!!!
    if do_save:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=False)
        fig_file_name = os.path.join(output_dir, "slices_prec_recall.pdf")

        plt.savefig(fig_file_name, bbox_inches='tight')
        plt.savefig(fig_file_name.replace(".pdf", ".jpeg"), bbox_inches='tight')
        print("INFO - Successfully saved fig {}".format(fig_file_name))
    if do_show:
        plt.show()

    # ------------------------------------------- figure 2 ------------------------------------------------

    # ROC curve for Slice Detection. x-axis: Specificity=False Positive rate, y-axis: recall/sensitivity
    # To compute average recall we basically compute the integral under the curve. 1) compute differences/steps
    # on the x-axis (specificity) and then multiply with the recall values.
    # Important: we need to extend the specificity values at the end (add 1) in order to obtain also a value for the
    # last x-axis value (
    specificity = copy.deepcopy(slice_fp_rate) + [1]
    specificity_diff = np.array(specificity[1:]) - np.array(specificity[:-1])
    average_recall = np.sum(np.array(slice_sensitivity) * specificity_diff)

    recall = [0] + copy.deepcopy(slice_sensitivity )
    specificity = [0] + copy.deepcopy(slice_fp_rate)
    ax = plt.figure(figsize=(width, height)).gca()
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(specificity, recall, color='b', alpha=0.2, where=step_kwargs['step'], label=model_tag)
    plt.fill_between(specificity, recall, alpha=0.2, color='b', **step_kwargs)
    # plt.plot(specificity, recall, alpha=0.5, color='b', linestyle='-')
    plt.ylabel('Recall', **config_detector.axis_font18)
    plt.xlabel('False positive rate (slices/volume)', **config_detector.axis_font18)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.tick_params(axis='both', which='major', labelsize=20)
    if model_tag is not None:
        plt.legend(loc="best", prop={'size': 18})
    plt.title('ROC-slices: AREC={:.2f}'.format(average_recall),
              **config_detector.axis_font18)
    if do_save:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=False)
        fig_file_name = os.path.join(output_dir, "slices_roc.pdf")

        plt.savefig(fig_file_name, bbox_inches='tight')
        plt.savefig(fig_file_name.replace(".pdf", ".jpeg"), bbox_inches='tight')
        print("INFO - Successfully saved fig {}".format(fig_file_name))
    if do_show:
        plt.show()
    # ------------------------------------------- figure 3 ------------------------------------------------

    average_precision = np.mean(slice_sensitivity)
    # for the computation of AUC of FROC we need to make sure slice_fp's are normalized between [0, 1]

    ax = plt.figure(figsize=(width, height)).gca()
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(slice_fp, slice_sensitivity, color='b', alpha=0.2, where=step_kwargs['step'], label=model_tag)
    plt.fill_between(slice_fp, slice_sensitivity, alpha=0.2, color='b', **step_kwargs)

    plt.ylabel('Recall/sensitivity', **config_detector.axis_font18)
    plt.xlabel('Avg #FP (slices/volume)', **config_detector.axis_font18)
    plt.ylim([0.0, 1.05])
    plt.xlim([np.min(slice_fp[slice_fp != 0]), np.max(slice_fp)])
    plt.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if model_tag is not None:
        plt.legend(loc="best", prop={'size': 18})
    plt.title('FROC-slices: AP={:.2f}'.format(average_precision),
              **config_detector.axis_font18)
    if do_save:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=False)
        fig_file_name = os.path.join(output_dir, "slices_froc.pdf")

        plt.savefig(fig_file_name, bbox_inches='tight')
        print("INFO - Successfully saved fig {}".format(fig_file_name))
    if do_show:
        plt.show()

    # ------------------------------------------- figure 4 ------------------------------------------------
    # FROC figure: y-axis -> detection rate
    #              x-axis -> #FP regions/volume
    average_detection_rate = np.mean(detection_rate)
    # for the computation of AUC of FROC we need to make sure slice_fp's are normalized between [0, 1]

    ax = plt.figure(figsize=(width, height)).gca()
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(region_fp, detection_rate, color='b', alpha=0.2, where=step_kwargs['step'], label=model_tag)
    plt.fill_between(region_fp, detection_rate, alpha=0.2, color='b', **step_kwargs)

    plt.ylabel('Detection rate volume', **config_detector.axis_font18)
    plt.xlabel(r'Avg #FP (regions/volume) (avg #regions$\approx$1700)', **config_detector.axis_font18)
    plt.xlim([0, 100])
    plt.ylim([0.0, 1.05])
    plt.tick_params(axis='both', which='major', labelsize=20)
    if model_tag is not None:
        plt.legend(loc="best", prop={'size': 18})
    plt.title('Voxel detection rate (mean per patient)',
              **config_detector.axis_font18)
    if do_save:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=False)
        fig_file_name = os.path.join(output_dir, "voxel_froc_detection_rate.pdf")

        plt.savefig(fig_file_name, bbox_inches='tight')
        plt.savefig(fig_file_name.replace(".pdf", ".jpeg"), bbox_inches='tight')
        print("INFO - Successfully saved fig {}".format(fig_file_name))
    if do_show:
        plt.show()
    plt.close()
