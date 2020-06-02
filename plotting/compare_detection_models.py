import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
from inspect import signature
from utils.common import translate_combined_model_tag

from networks.detection.general_setup import config_detector

PLOT_COLORS = ['xkcd:black',  'xkcd:blue', 'xkcd:bluish green', 'xkcd:sky blue',
               'xkcd:vermillion', 'xkcd:reddish purple']

ALPHA = 0.4
LW = 2.5
LINESTYLES = {'drn_mc': "--", 'dcnn_mc': "-.", 'unet_mc': ":",
              'drn_mc_ce': "--", 'dcnn_mc_brier': "-.", 'unet_mc_ce': ":",
              'drn_mc_dice': "--", 'dcnn_mc_dice': "-.", 'unet_mc_dice': ":"}

LARGE_FONT_SIZE = {'fontname': 'Monospace', 'size': '28', 'color': 'black', 'weight': 'normal'}


def get_descr_umap(io_channel):
    return 'e-map' if io_channel == 'emap' else 'b-map'


def compare_pr_rec_curves(result_dict, x_measure, y_measure,
                          file_prefix=None, do_show=True, do_save=False,
                          width=9, height=9, output_dir=None):
    step_kwargs = ({'step': 'pre'} if 'step' in signature(plt.fill_between).parameters else {})

    _ = plt.figure(figsize=(width, height))

    plot_num = 220
    for c, io_channel in enumerate(['emap', 'bmap']):
        umap = get_descr_umap(io_channel)
        plt.subplot(plot_num + c + 1)
        for j, model_label in enumerate(result_dict.keys()):
            recall = result_dict[model_label][io_channel][x_measure]
            precision = result_dict[model_label][io_channel][y_measure]
            # Make sure first recall value is always 0 (and prec = 1)

            recall = np.array([0] + recall)
            precision = np.array([1] + precision)
            rec_diff = np.array(recall[1:]) - np.array(recall[:-1])
            average_precision = np.sum(np.array(precision[1:]) * rec_diff)
            # plt.plot(recall, precision, lw=4, alpha=0.3, color=plot_colors[j],
            #          label="{} AP={:.2f}".format(model_label, average_precision))
            plt.step(recall, precision, lw=LW, alpha=ALPHA, color=PLOT_COLORS[j], where=step_kwargs['step'],
                     label="{} (AP={:.2f})".format(translate_combined_model_tag(model_label), average_precision))
            # linestyle=LINESTYLES[model_label]
        plt.plot([0, 1], [1, 0], color='navy', alpha=ALPHA, lw=LW, linestyle='--')
        plt.xlabel('Recall', **LARGE_FONT_SIZE)
        plt.ylabel('Precision', **LARGE_FONT_SIZE)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.title('{}'.format(umap),
                  **LARGE_FONT_SIZE)
        plt.legend(loc="best", prop={'size': 20})
    plt.tight_layout(rect=[0.04, 0.04, 0.96, 0.96])

    if do_save:
        if file_prefix is not None:
            fname = file_prefix + "_slices_prec_recall.pdf"
        else:
            fname = "slices_prec_recall.pdf"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=False)
        fig_file_name = os.path.join(output_dir, fname)
        plt.savefig(fig_file_name, bbox_inches='tight')
        plt.savefig(fig_file_name.replace(".pdf", ".png"), bbox_inches='tight')
        print("INFO - Successfully saved fig {}".format(fig_file_name))

    if do_show:
        plt.show()
    plt.close()


def compare_roc_curves(result_dict, x_measure, y_measure, model_labels,
                          file_prefix=None, do_show=True, do_save=False,
                          width=9, height=9, output_dir=None):

    _ = plt.figure(figsize=(width, height))

    plot_num = 220
    for c, io_channel in enumerate(['emap', 'bmap']):
        plt.subplot(plot_num + c + 1)
        for j, model_label in enumerate(result_dict.keys()):
            recall = result_dict[model_label][io_channel][x_measure]
            specificity = result_dict[model_label][io_channel][y_measure]

            # Make sure first recall value is always 0 (and prec = 1)
            recall = np.array([0] + recall)
            specificity = np.array(specificity + [1])
            rec_diff = np.array(recall[1:]) - np.array(recall[:-1])
            average_recall = np.sum(np.array(recall[1:]) * rec_diff)
            plt.plot(specificity, recall, lw=LW, alpha=ALPHA, color=PLOT_COLORS[j],
                     label="{} (avg-rec={:.2f})".format(translate_combined_model_tag(model_label), average_recall))

        plt.plot([0, 1], [1, 0], color='navy', alpha=ALPHA, lw=LW, linestyle='--')
        plt.xlabel('Specificity', **config_detector.axis_font18)
        plt.ylabel('Recall', **config_detector.axis_font18)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.title('{}: ROC slice detection'.format(io_channel),
                  **config_detector.axis_font18)
        plt.legend(loc="best", prop={'size': 20})
    plt.tight_layout(rect=[0.04, 0.04, 0.96, 0.96])

    if do_save:
        if file_prefix is not None:
            fname = file_prefix + "_slices_roc.pdf"
        else:
            fname = "slices_roc.pdf"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=False)
        fig_file_name = os.path.join(output_dir, fname)
        plt.savefig(fig_file_name, bbox_inches='tight')
        plt.savefig(fig_file_name.replace(".pdf", ".png"), bbox_inches='tight')
        print("INFO - Successfully saved fig {}".format(fig_file_name))

    if do_show:
        plt.show()
    plt.close()


def compare_froc_curves_slices(result_dict, x_measure, y_measure, file_prefix=None,
                               do_show=True, do_save=False, width=10, height=8, output_dir=None):
    _ = plt.figure(figsize=(width, height)).gca()

    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})

    plot_num = 220
    for c, io_channel in enumerate(['emap', 'bmap']):
        umap = get_descr_umap(io_channel)
        ax = plt.subplot(plot_num + c + 1)
        min_slice_fp, max_slice_fp = 1000, -1
        for j, model_label in enumerate(result_dict.keys()):
            slice_fp = result_dict[model_label][io_channel][x_measure]
            recall = result_dict[model_label][io_channel][y_measure]
            average_recall = np.mean(recall)
            # plt.plot(slice_fp, recall, lw=4, alpha=0.3, color=plot_colors[j],
            #          label="{}".format(model_label))
            plt.step(slice_fp, recall, lw=LW, alpha=ALPHA, color=PLOT_COLORS[j], where=step_kwargs['step'],
                     label="{}".format(translate_combined_model_tag(model_label)))
            m, mx = np.min(slice_fp[slice_fp != 0]), np.max(slice_fp)
            min_slice_fp = m if m < min_slice_fp else min_slice_fp
            max_slice_fp = mx if mx > max_slice_fp else max_slice_fp

        plt.xlabel('#FP slices/volume', **config_detector.axis_font18)
        plt.ylabel('Sensitivity', **config_detector.axis_font18)
        plt.ylim([0.0, 1.05])
        plt.xlim([0, max_slice_fp])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.title('{}'.format(umap),
                  **config_detector.axis_font18)
        plt.legend(loc="best", prop={'size': 20})
    plt.tight_layout(rect=[0.04, 0.04, 0.96, 0.96])

    if do_save:
        if file_prefix is not None:
            fname = file_prefix + "_froc_slice_detection.pdf"
        else:
            fname = "froc_slice_detection.pdf"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=False)
        fig_file_name = os.path.join(output_dir, fname)

        plt.savefig(fig_file_name, bbox_inches='tight')
        plt.savefig(fig_file_name.replace(".pdf", ".png"), bbox_inches='tight')
        print("INFO - Successfully saved fig {}".format(fig_file_name))

    if do_show:
        plt.show()
    plt.close()


def compare_froc_curves_dtrate(result_dict, x_measure, y_measure,
                               file_prefix=None,
                               do_show=True, do_save=False, width=10, height=8, output_dir=None):
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})

    ax = plt.figure(figsize=(width, height)).gca()

    plot_num = 220
    for c, io_channel in enumerate(['emap', 'bmap']):
        umap = get_descr_umap(io_channel)
        plt.subplot(plot_num + c + 1)
        for j, model_label in enumerate(result_dict.keys()):
            region_fp = result_dict[model_label][io_channel][x_measure]
            detection_rate = result_dict[model_label][io_channel][y_measure]
            average_dt = np.mean(detection_rate)
            # plt.plot(region_fp, detection_rate, lw=4, alpha=0.3, color=plot_colors[j],
            #          label="{}".format(model_label))
            plt.step(region_fp, detection_rate, lw=LW, alpha=ALPHA, color=PLOT_COLORS[j], where=step_kwargs['step'],
                     label="{}".format(translate_combined_model_tag(model_label)))

        plt.xlabel('#FP regions/volume', **LARGE_FONT_SIZE)
        plt.ylabel('Sensitivity', **LARGE_FONT_SIZE)
        plt.ylim([0.0, 1.05])
        plt.xlim([0, 120])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.title('{}'.format(umap),
                  **LARGE_FONT_SIZE)
        plt.legend(loc="best", prop={'size': 20})

    plt.tight_layout(rect=[0.04, 0.04, 0.96, 0.96])
    if do_save:
        if file_prefix is not None:
            fname = file_prefix + "_froc_voxel_detection_rate.pdf"
        else:
            fname = "froc_voxel_detection_rate.pdf"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=False)
        fig_file_name = os.path.join(output_dir, fname)

        plt.savefig(fig_file_name, bbox_inches='tight')
        plt.savefig(fig_file_name.replace(".pdf", ".png"), bbox_inches='tight')
        print("INFO - Successfully saved fig {}".format(fig_file_name))

    if do_show:
        plt.show()
    plt.close()