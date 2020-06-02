
import matplotlib.pyplot as plt
from plotting.color_maps import SegColorMap
import copy
from plotting.settings import plot_settings

import numpy as np
import os
from matplotlib import cm


def center_crop(img_slice, size=64):
    w, h = img_slice.shape
    x, y = int(w / 2), int(h / 2)
    if x < size:
        x = 0
    else:
        x -= size
    if y < size:
        y = 0
    else:
        y -= size
    return img_slice[x:x+int(2*size), y:y+int(size*2)]


def plot_slices_per_phase(data_handler, patient_id, do_show=True, do_save=False, do_crop=False,
                          left_column_overlay="map", alpha=0.5, cardiac_phase="ES", threshold=0.5,
                          slice_range=None, type_of_map="e_map", right_column_overlay="error"):

    """

    :param data_handler:
    :param patient_id:
    :param do_show:
    :param do_save:
    :param slice_range:  e.g. [0, 5]
    :param type_of_map: e_map or b_map
    :param left_column_overlay: "map" = uncertainty maps or
                                "error_roi" = seg error regions that needs to be detected
                                "heat_map" = region detection heat map
    :param right_column_overlay: ["ref", "error", "auto"] type of segmentation mask that we plot in the right figure
    :return:
    """

    def transparent_cmap(cmap, alpha=0.8, N=255):
        """ Copy colormap and set alpha values """
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.linspace(0, alpha, N + 4)
        return mycmap

    if right_column_overlay not in [None, "ref", "error", "auto", "heat_map", "umap", "error_roi"]:
        raise ValueError("ERROR - right_column_overlay must be ref, error, heat_map, error_roi or auto! "
                         "(and not {})".format(right_column_overlay))

    if left_column_overlay not in [None, "ref", "error", "auto", "heat_map", "umap", "error_roi"]:
        raise ValueError("ERROR - left_column_overlay must be ref, error, heat_map, error_roi or auto! "
                         "(and not {})".format(left_column_overlay))

    if type_of_map not in ["e_map", "b_map"]:
        raise ValueError("ERROR - type_of_map must be e_map or b_map! (and not {})".format(type_of_map))
    # Use base cmap to create transparent
    mycmap = transparent_cmap(plt.get_cmap('jet'))
    mycmap_plasma = transparent_cmap(plt.get_cmap('plasma'), alpha=alpha)
    my_seg_mag_trans = SegColorMap(alpha=alpha)

    map_max = 1.0
    umap = data_handler["umaps"][patient_id]['umap']
    pred_labels = data_handler["pred_labels"][patient_id]

    if left_column_overlay == "heat_map" or right_column_overlay == "heat_map":
        heat_map = copy.deepcopy(data_handler['heat_maps'][patient_id])
        heat_map[heat_map < threshold] = 0
    else:
        heat_map = None

    mri_image = data_handler['images'][patient_id]
    # mri_image = mri_image[config_acdc.pad_size:-config_acdc.pad_size, config_acdc.pad_size:-config_acdc.pad_size]
    labels = data_handler['ref_labels'][patient_id]

    if right_column_overlay == "error_roi" or left_column_overlay == "error_roi":
        # has shape [z, nclasses, y, x]
        errors_to_detect = data_handler['dt_labels'][patient_id]

    else:
        errors_to_detect = None

    seg_error_volume = data_handler['pred_labels'][patient_id] != data_handler['ref_labels'][patient_id]

    if slice_range is None:
        num_of_slices = umap.shape[0]
        str_slice_range = "_s1_" + str(num_of_slices)
        slice_range = np.arange(num_of_slices)
    else:
        str_slice_range = "_s" + "_".join([str(i) for i in slice_range])
        slice_range = np.arange(slice_range[0], slice_range[1])
        num_of_slices = len(slice_range)

    columns = 4
    width = 16
    height = 10 * num_of_slices
    row = 0

    fig = plt.figure(figsize=(width, height))
    fig.suptitle(patient_id, **plot_settings.title_font_medium)
    for slice_id in slice_range:
        rows = 2 * num_of_slices
        # print(umap.shape, pred_labels.shape, labels.shape, mri_image.shape)
        umap_slice = umap[slice_id, :, :]
        umap_slice[umap_slice < 0] = 0
        umap_slice = center_crop(umap_slice) if do_crop else umap_slice
        img_slice = mri_image[slice_id, :, :]
        img_slice = center_crop(img_slice) if do_crop else img_slice
        u_min, u_max = np.min(umap_slice), np.max(umap_slice)

        labels_slice = labels[slice_id, :, :]
        labels_slice = center_crop(labels_slice) if do_crop else labels_slice
        # IMPORTANT: (already verified) Assuming labels AND pred_labels has shape [8, w, h, #slices]
        pred_labels_slice = pred_labels[slice_id, :, :]
        pred_labels_slice = center_crop(pred_labels_slice) if do_crop else pred_labels_slice
        # errors_slice = detect_seg_errors(labels_slice, pred_labels_slice, is_multi_class=False)
        errors_slice = seg_error_volume[slice_id, :, :]
        errors_slice = center_crop(errors_slice) if do_crop else errors_slice
        if heat_map is not None:
            heat_map_slice = heat_map[slice_id, :, :]
            heat_map_slice = center_crop(heat_map_slice) if do_crop else heat_map_slice
        if right_column_overlay == "error_roi" or left_column_overlay == "error_roi":
            errors_slice_to_detect = np.any(errors_to_detect[slice_id], axis=0).astype(np.int)
            errors_slice_to_detect = center_crop(errors_slice_to_detect) if do_crop else errors_slice_to_detect

        ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)

        ax1.imshow(img_slice, cmap=cm.gray)

        if left_column_overlay == "umap":
            ax1plot = ax1.imshow(umap_slice, cmap=mycmap_plasma, vmin=0, vmax=1.)  # 0.0001
            # ax1.set_aspect('auto')
            # fig.colorbar(ax1plot, ax=ax1, fraction=0.046, pad=0.04)
            left_title_suffix = " (uncertainties)"
        elif left_column_overlay == "error_roi":
            # errors_slice_to_detect = convert_to_multiclass(errors_slice_to_detect)
            error_count = np.count_nonzero(errors_slice_to_detect)
            _ = ax1.imshow(errors_slice_to_detect, cmap=my_seg_mag_trans.cmap)
            left_title_suffix = " (error rois # {})".format(error_count)
        elif left_column_overlay == "heat_map":
            _ = ax1.imshow(heat_map_slice, cmap=mycmap_plasma)
            left_title_suffix = " (heat maps)"
        elif left_column_overlay == "ref":
            multi_ref_labels = my_seg_mag_trans.convert_multi_labels(labels_slice)
            ax1.imshow(multi_ref_labels, cmap=my_seg_mag_trans.cmap)
            left_title_suffix = " (ref)"
        else:
            left_title_suffix = ""

        plt.axis("off")
        p_title = "{} {} slice {}: ".format(type_of_map, cardiac_phase, slice_id + 1)

        ax1.set_title(p_title + left_title_suffix, **plot_settings.title_font_small)
        ax2 = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
        ax2.imshow(img_slice, cmap=cm.gray)
        # What do we plot in the right column?
        if right_column_overlay == "ref":
            # multi_label_slice = convert_to_multiclass(labels_slice)
            ax2.imshow(labels_slice, cmap=mycmap)
            ax2.set_title("Reference (r=LV/y=myo/b=RV)", **plot_settings.title_font_small)
        elif right_column_overlay == "error":
            copy_pred_labels_slice = copy.deepcopy(pred_labels_slice)
            copy_pred_labels_slice[errors_slice != 0] = 5
            multi_pred_labels = my_seg_mag_trans.convert_multi_labels(copy_pred_labels_slice)
            ax2.imshow(multi_pred_labels, cmap=my_seg_mag_trans.cmap)
            error_count = np.count_nonzero(errors_slice)
            ax2.set_title("Segmentation errors (# {}) (r=LV/y=myo/b=RV)".format(error_count),
                          **plot_settings.title_font_small)
        elif right_column_overlay == "error_roi":
            error_count = np.count_nonzero(errors_slice_to_detect)
            _ = ax2.imshow(errors_slice_to_detect, cmap=mycmap)
            ax2.set_title("Region detection error rois (# {})".format(error_count), **plot_settings.title_font_small)
        elif right_column_overlay == "heat_map":
            _ = ax2.imshow(heat_map_slice, cmap=mycmap)
            ax2.set_title("Heat maps region detection", **plot_settings.title_font_small)
        elif right_column_overlay == "umap":
            _ = ax2.imshow(umap_slice, cmap=mycmap, vmin=0., vmax=map_max)
            ax2.set_title("Uncertainties", **plot_settings.title_font_small)
        else:
            # automatic seg-mask
            multi_pred_labels = my_seg_mag_trans.convert_multi_labels(pred_labels_slice)
            ax2.imshow(multi_pred_labels, cmap=my_seg_mag_trans.cmap)
            ax2.set_title("Automatic mask (r=LV/y=myo/b=RV)", **plot_settings.title_font_small)
        plt.axis("off")
        row += 2

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:

        fig_path = os.path.expanduser(os.path.join(data_handler["src_data_path"], 'figures'))
        fig_path = os.path.join(fig_path, patient_id)
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        fig_name = patient_id + "_" + type_of_map
        fig_name += str_slice_range
        fig_name = os.path.join(fig_path, fig_name + ".pdf")

        plt.savefig(fig_name, bbox_inches='tight')
        print(("INFO - Successfully saved fig %s" % fig_name))
    if do_show:
        plt.show()

