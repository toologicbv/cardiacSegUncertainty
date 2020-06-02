import glob
import os
import numpy as np
import SimpleITK as sitk
from utils.common import one_hot_encoding
from datasets.common import apply_2d_zoom_3d
from datasets.ACDC.data import get_acdc_patient_ids, ACDCImage
from datasets.common import get_config
from utils.detection.heat_map import ImageHeatMapHandler


OUTPUT_DIRS = {"pred_probs": "pred_probs", "pred_labels": "pred_labels", "umaps": "umaps", "dt_maps": "dt_maps",
               "dt_labels": "dt_labels", "heat_maps": "heat_maps"}
mc_suffix = "_mc"
file_ext = ".nii.gz"


def get_search_mask(src_data_path, load_type, cardiac_phase, mc_dropout=False, patient_id=None):
    assert load_type in ['pred_probs', 'pred_labels', 'dt_maps', 'dt_labels', 'heat_maps']
    output_dir = OUTPUT_DIRS[load_type]
    if mc_dropout:
        input_dir = output_dir + mc_suffix
    else:
        input_dir = output_dir
    if patient_id is None:
        input_dir += os.path.sep + "*_{}".format(cardiac_phase) + file_ext
    else:
        input_dir += os.path.sep + patient_id + "_{}".format(cardiac_phase) + file_ext

    return os.path.join(src_data_path, input_dir)


def load_data(src_data_path, cardiac_phase, type_of_data, mc_dropout=False, dt_config_id=None, patient_id=None,
              one_hot=True, meta_info=False):
    """

    :param src_data_path:
    :param cardiac_phase:
    :param type_of_data: is a list with possible strings: pred_labels, pred_probs, ref_labels
                         AND NOTE THAT ref_labels is never the only data type we're loading and
                         others are preceding ref_labels (so can be never the first, because we don't pass
                         the fold id and only load ref_labels of patients that we have loaded before)
    :param mc_dropout:
    :param dt_config_id: applicable for loading dt maps and dt labels. Specifies the margins for the
            segmentation inaccuracies e.g. "fixed_42_28"
    :param patient_id: e.g. "patient010" if none all patient files in src_data_path are loaded, otherwise only
            a specific patient
    :param meta_info: boolean. If True for pred_labels and pred_probs the get-function returns a dictionary
                        containing the numpy data and the spacing ([z, y, x])
    :return:
    """
    assert type_of_data[0] != "ref_labels"
    src_data_path = os.path.expanduser(src_data_path)
    data_dict = {'src_data_path': src_data_path, 'mc_dropout': mc_dropout,
                 'meta_info': meta_info, 'cardiac_phase': cardiac_phase}
    if not isinstance(type_of_data, list):
        type_of_data = [type_of_data]

    for data_type in type_of_data:

        if data_type == "pred_labels":

            data_dict[data_type] = get_pred_labels(src_data_path, cardiac_phase, patient_id=patient_id,
                                                       mc_dropout=mc_dropout, one_hot=one_hot, meta_info=meta_info)
        elif data_type == "pred_probs":
            data_dict[data_type] = get_pred_probs(src_data_path, cardiac_phase, patient_id=patient_id,
                                                       mc_dropout=mc_dropout, meta_info=meta_info)

        elif data_type == "dt_maps":
            assert dt_config_id is not None
            data_dict[data_type] = get_dt_maps(src_data_path, cardiac_phase, dt_config_id, patient_id=patient_id)

        elif data_type == "dt_labels":
            assert dt_config_id is not None
            data_dict[data_type] = get_detector_target_labels(src_data_path, cardiac_phase, dt_config_id,
                                                              patient_id=patient_id, mc_dropout=mc_dropout)

        elif data_type == "umaps":
            data_dict[data_type] = get_umaps(src_data_path, cardiac_phase, mc_dropout, patient_id=patient_id)
        elif data_type == "ref_labels":
            ref_labels = dict()
            apex_base_slices = dict()
            # IMPORTANT: we assume that type_of_data is a list where first entry is never ref_labels
            for patid in data_dict[type_of_data[0]].keys():
                # although we pass one patient_id the function returns a dict
                gt_labels, a_b_slices = get_ref_labels(cardiac_phase=cardiac_phase, patient_id=patid,
                                                                                      one_hot=one_hot,
                                                                                      limited_load=False,
                                                                                      resample=False)
                ref_labels[patid], apex_base_slices[patid] = gt_labels[patid], a_b_slices[patid]
            data_dict[data_type] = ref_labels
            data_dict['apex_base_slices'] = apex_base_slices

        elif data_type == "images":
            images = dict()
            apex_base_slices = dict()
            # IMPORTANT: we assume that type_of_data is a list where first entry is never ref_labels
            for patid in data_dict[type_of_data[0]].keys():
                # although we pass one patient_id the function returns a dict
                img_dict, a_b_slices = get_images(cardiac_phase=cardiac_phase, patient_id=patid,
                                                                                      limited_load=False,
                                                                                      resample=False)
                images[patid], apex_base_slices[patid] = img_dict[patid], a_b_slices[patid]
            data_dict[data_type] = images
            data_dict['apex_base_slices'] = apex_base_slices

        elif data_type == "heat_maps":
            heat_dict = get_heat_maps(src_data_path, cardiac_phase=cardiac_phase, patient_id=patient_id, mc_dropout=mc_dropout)
            # unfortunately the ImageHeatMapHandler returns dictionary with two keys patient_id and cardiac_phase. We don't change
            # that implementation but make sure we here omit the cardiac_phase dictionary key
            temp_dict = {}
            for p_id in heat_dict.keys():
                temp_dict[p_id] = heat_dict[p_id][cardiac_phase]
            data_dict[data_type] = temp_dict
            del temp_dict
        else:
            raise ValueError("ERROR - load data - unknown data type >> {} <<".format(data_type))

    return data_dict


def get_ref_labels(fold=None, dataset=None, cardiac_phase="ES", patient_id=None, one_hot=False, limited_load=False,
                   resample=False):
    dta_settings = get_config('ACDC')
    if patient_id is None:
        assert fold is not None and dataset is not None
        pat_nums = get_acdc_patient_ids(fold, dataset, limited_load=limited_load)
    else:
        pat_nums = [int(patient_id.strip("patient"))]
    ref_labels = dict()
    apex_base_slices = dict()
    for patnum in pat_nums:

        img = ACDCImage(patnum, root_dir=dta_settings.short_axis_dir, resample=resample, scale_intensities=True)
        if cardiac_phase == 'ED':
            image, sp, reference = img.ed()
            a_b_slices = img.base_apex_slice_ed
        else:
            image, sp, reference = img.es()
            a_b_slices = img.base_apex_slice_es
        if one_hot:
            reference = one_hot_encoding(reference)
        ref_labels[img.patient_id] = reference
        apex_base_slices[img.patient_id] = a_b_slices

    return ref_labels, apex_base_slices


def get_images(fold=None, dataset=None, cardiac_phase="ES", patient_id=None, limited_load=False,
                   resample=False):

    dta_settings = get_config('ACDC')
    if patient_id is None:
        assert fold is not None and dataset is not None
        pat_nums = get_acdc_patient_ids(fold, dataset, limited_load=limited_load)
    else:
        pat_nums = [int(patient_id.strip("patient"))]
    images = dict()
    apex_base_slices = dict()
    for patnum in pat_nums:

        img = ACDCImage(patnum, root_dir=dta_settings.short_axis_dir, resample=resample, scale_intensities=True)
        if cardiac_phase == 'ED':
            image, sp, reference = img.ed()
            a_b_slices = img.base_apex_slice_ed
        else:
            image, sp, reference = img.es()
            a_b_slices = img.base_apex_slice_es

        images[img.patient_id] = image
        apex_base_slices[img.patient_id] = a_b_slices

    return images, apex_base_slices


def get_pred_labels(src_data_path, cardiac_phase, mc_dropout=False, patient_id=None,
                    one_hot=False, meta_info=True):
    """
    returns predicted labels as numpy array of shape [z, y, x] and spacing
    """

    search_path = get_search_mask(src_data_path, "pred_labels", cardiac_phase, mc_dropout, patient_id)

    pred_labels = {}
    file_list = glob.glob(search_path)
    if len(file_list) == 0:
        raise ValueError("ERROR - No search result for {}".format(search_path))
    file_list.sort()

    for fname in file_list:
        pat_id, _ = (os.path.splitext(os.path.basename(fname))[0]).split("_")

        img = sitk.ReadImage(fname)
        org_sp = img.GetSpacing()[::-1]
        predictions = sitk.GetArrayFromImage(img).astype(np.long)
        # We store pred labels (and uncertainty maps) in the original spacing (according to the image info).
        # Nevertheless, in case the spacing of x, y is below 1mm we resample to 1.4 isotropic. We do the same
        # in the ACDC dataset object
        if org_sp[-1] < 1.:
            spacing = np.array([org_sp[0], 1.4, 1.4]).astype(np.float64)
            predictions = apply_2d_zoom_3d(predictions, org_sp, order=0, do_blur=False, as_type=np.int,
                                           new_spacing=spacing)
        else:
            spacing = org_sp

        if one_hot:
            predictions = one_hot_encoding(predictions)
        if meta_info:
            pred_labels[pat_id] = {'pred_labels': predictions, 'spacing': spacing, 'original_spacing': org_sp}
        else:
            pred_labels[pat_id] = predictions
        del predictions

    return pred_labels


def get_umaps(src_data_path, cardiac_phase, mc_dropout=False, patient_id=None):
    """
    returns uncertainty map (e- or b-map) as numpy array of shape [z, y, x] and spacing
    """

    if mc_dropout:
        input_dir = "_bmap" + file_ext
    else:
        input_dir = "_emap" + file_ext
    if patient_id is None:
        input_dir = os.path.sep + "*_{}".format(cardiac_phase) + input_dir
    else:
        input_dir = os.path.sep + patient_id + "_{}".format(cardiac_phase) + input_dir
    search_path = src_data_path + os.sep + OUTPUT_DIRS['umaps'] + os.sep + input_dir

    umaps = {}
    file_list = glob.glob(search_path)
    if len(file_list) == 0:
        raise ValueError("ERROR - No search result for {}".format(search_path))
    file_list.sort()

    for fname in file_list:
        pat_id, _, _ = (os.path.splitext(os.path.basename(fname))[0]).split("_")

        img = sitk.ReadImage(fname)
        org_sp = img.GetSpacing()[::-1]
        data = sitk.GetArrayFromImage(img).astype(np.float32)
        # hacky!!! For entropy maps it happens that some voxels have tiny negative values e-08.
        # But we need to make sure these are zero
        # data[data < 0] = 0
        # We store pred labels (and uncertainty maps) in the original spacing (according to the image info).
        # Nevertheless, in case the spacing of x, y is below 1mm we resample to 1.4 isotropic. We do the same
        # in the ACDC dataset object
        if org_sp[-1] < 1.:
            spacing = np.array([org_sp[0], 1.4, 1.4]).astype(np.float64)
            data = apply_2d_zoom_3d(data, org_sp, do_blur=True, new_spacing=spacing)
        else:
            spacing = org_sp
        umaps[pat_id] = {'umap': data, 'spacing': spacing, 'original_spacing': org_sp}
        del data

    return umaps


def get_pred_probs(src_data_path, cardiac_phase, mc_dropout=False, patient_id=None, meta_info=True):
    """
    returns softmax probabilities as numpy array of shape [z, nclasses, y, x] and spacing
    """

    search_path = get_search_mask(src_data_path, "pred_probs", cardiac_phase, mc_dropout, patient_id)

    pred_probs = {}
    file_list = glob.glob(search_path)
    if len(file_list) == 0:
        raise ValueError("ERROR - No search result for {}".format(search_path))
    file_list.sort()

    for fname in file_list:
        pat_id, _ = (os.path.splitext(os.path.basename(fname))[0]).split("_")
        img = sitk.ReadImage(fname)
        org_sp = img.GetSpacing()[::-1]
        softmax_probs = sitk.GetArrayFromImage(img).astype(np.float32)

        # We store pred probs in the original spacing (according to the image info).
        # Nevertheless, in case the spacing of x, y is below 1mm we resample to 1.4 isotropic. We do the same
        # in the ACDC dataset object
        if org_sp[-1] < 1.:
            nclasses = softmax_probs.shape[1]
            t_sp = np.array([org_sp[0], 1.4, 1.4]).astype(np.float64)
            t_org_sp = np.array([org_sp[0], org_sp[2], org_sp[3]]).astype(np.float64)
            spacing = np.array([org_sp[0], org_sp[1], 1.4, 1.4]).astype(np.float64)
            resampled_probs = []
            for cls_idx in np.arange(nclasses):
                resampled_probs.append(apply_2d_zoom_3d(softmax_probs[:, cls_idx].astype(np.float32), t_org_sp,
                                                        do_blur=False, new_spacing=t_sp))
            softmax_probs = np.concatenate(([arr[:, None] for arr in resampled_probs]), axis=1)

        else:
            spacing = org_sp

        if meta_info:
            pred_probs[pat_id] = {'pred_probs': softmax_probs, 'spacing': spacing, 'original_spacing': org_sp}
        else:
            pred_probs[pat_id] = softmax_probs
        del softmax_probs

    return pred_probs


def get_heat_maps(src_data_path, cardiac_phase, patient_id=None, mc_dropout=False):
    heat_map_dir = os.path.join(src_data_path, OUTPUT_DIRS["heat_maps"])
    heat_map_handler = ImageHeatMapHandler(mode_dataset="separate")
    if patient_id is None:
        heat_map_handler.load(heat_map_dir, cardiac_phase,  mc_dropout=mc_dropout)

    else:
        heat_map_handler.load_heat_map(heat_map_dir, patient_id, cardiac_phase, mc_dropout=mc_dropout)
    return heat_map_handler.heat_maps


def get_dt_maps(src_data_path, cardiac_phase, dt_config_id, patient_id=None):

    search_mask = os.path.join(os.path.join(src_data_path, OUTPUT_DIRS['dt_maps'], dt_config_id))
    if patient_id is None:
        search_mask += os.path.sep + "*_{}".format(cardiac_phase) + ".npz"
    else:
        search_mask += os.path.sep + patient_id + "_{}".format(cardiac_phase) + ".npz"

    dt_maps = {}
    file_list = glob.glob(search_mask)
    if len(file_list) == 0:
        raise ValueError("ERROR - No search result for {}".format(search_mask))
    file_list.sort()

    for fname in file_list:
        pat_id, _ = (os.path.splitext(os.path.basename(fname))[0]).split("_")
        data = np.load(fname)
        dt_maps[pat_id] = data['dt_map']

    return dt_maps


def get_detector_target_labels(src_data_path, cardiac_phase, dt_config_id, patient_id=None, mc_dropout=False):

    search_mask = os.path.join(os.path.join(src_data_path, OUTPUT_DIRS['dt_labels'], dt_config_id))
    search_suffix = "_mc.npz" if mc_dropout else ".npz"
    if patient_id is None:
        search_mask += os.path.sep + "*_{}".format(cardiac_phase) + search_suffix
    else:
        search_mask += os.path.sep + patient_id + "_{}".format(cardiac_phase) + search_suffix

    dt_labels = {}
    file_list = glob.glob(search_mask)
    if len(file_list) == 0:
        raise ValueError("ERROR - No search result for {}".format(search_mask))
    file_list.sort()

    for fname in file_list:
        if mc_dropout:
            pat_id, _, _ = (os.path.splitext(os.path.basename(fname))[0]).split("_")
        else:
            pat_id, _ = (os.path.splitext(os.path.basename(fname))[0]).split("_")
        data = np.load(fname)
        dt_labels[pat_id] = data['dt_ref_labels']

    return dt_labels


if __name__ == "__main__":
    # mydict = get_umaps("/home/jorg/expers/bayes_drn_brier", 'ES', mc_dropout=False)
    # print(mydict.keys())
    # print(mydict['patient016']['umap'].shape, mydict['patient016']['spacing'])

    dt_maps = get_detector_target_labels("/home/jorg/expers/acdc/drn_mc_ce", "ES", "fixed_42_28")
    for pat, dt in dt_maps.items():
        print(pat, dt.shape)
