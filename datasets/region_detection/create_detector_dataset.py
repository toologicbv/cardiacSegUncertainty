
from tqdm import tqdm
import numpy as np

from datasets.ACDC.data import get_acdc_patient_ids
from datasets.ACDC.get_data import load_data
from datasets.region_detection.detector_dataset import RegionDetectorDataSet
from datasets.ACDC.data import ACDCImage
from datasets.common import get_config

dta_settings = get_config('ACDC')


def create_label_contours(p_labels):
    """

    :param p_labels: is numpy shape of [z, y, x] with predicted multi-labels {0...3}
    :return:
    """
    from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
    contours = np.zeros_like(p_labels)
    footprint = generate_binary_structure(2, 1)
    classes = np.unique(p_labels)
    for slice_id in np.arange(p_labels.shape[0]):
        p_slice = p_labels[slice_id]
        for cls_idx in classes:
            if cls_idx != 0:
                mask = np.logical_xor(p_slice == cls_idx, binary_erosion(p_slice == cls_idx, structure=footprint, iterations=1))
                contours[slice_id, mask] = 1

    return contours


def create_dataset(fold, src_data_path, mc_dropout=False, num_of_input_chnls=3, limited_load=False,
                   dt_config_id=None, cardiac_phases=tuple(('ES', 'ED')), test_only=False):
    """

    :param fold: This is the fold_id we use for the experiment handler
    :param src_data_path: absolute path to directory that holds dirs for
                            dt_labels, pred_labels.
                          We assume all folds are in this directory.

    :param dt_config_id: IMPORTANT: if None we use dt_maps, target_labels from ~/data/ACDC/brier/ root directory
                         and otherwise we load these things from root-directory/dt_config_id/
                         This enables us to separate runs between different low-quality filtering approaches

    :param mc_dropout: True: Bayesian uncertainty map; False: Entropy maps;
    :param num_of_input_chnls: 2: automatic segmentation masks + uncertainties; 3: + original image
    :param limited_load: reduce training dataset to small number (currently 10)
    :param model_name: please refer to the model parameter in the parsing.py file to determine which argument values
                        are currently valid (rd1, rd2, rd3)

    :param cardiac_phases:
    :return: RegionDetector object containing data set
    """
    # only works for fold0 (testing purposes, small dataset), we can overwrite the quick_run argument
    # means we're loading a very small dataset (probably 2-3 images)
    # get the patient ids of the test set for this segmentation handler. We use that to distinguish between
    # train and test set patient ids.
    # reduce training set in case we are debugging the model
    test_ids = get_acdc_patient_ids(fold, "validation", limited_load=limited_load)
    train_ids = get_acdc_patient_ids(fold, "training", limited_load=limited_load)

    print("INFO - Preparing ACDC data handler. This may take a while. Be patient...")
    # REMEMBER: type of map determines whether we're loading mc-dropout predictions or single-predictions
    # instead of using class labels 0, 1, 2, 3 for the seg-masks we will use values between [0, 1]
    # 19-11-2018 Changed the labels back to [1...4] instead of [0, 0.3, 0.6, 0.9] because the former yields slightly
    # better validation performance
    labels_float = [1., 2., 3., 4.]
    dataset = RegionDetectorDataSet(num_of_channels=num_of_input_chnls, mc_dropout=mc_dropout)
    if test_only:
        patient_ids = test_ids
    else:
        patient_ids = np.concatenate((train_ids, test_ids), axis=0)
    type_of_data = ["pred_labels", "dt_labels", "umaps"]
    # loop over training images
    # patient_ids = [82]
    for patid in tqdm(patient_ids):
        patient_id = "patient{:03d}".format(patid)
        # we store the list indices for this patient in RegionDetector.train_images or test_images
        # this can help us for plotting the dataset or batches (debugging)
        for cardiac_phase in cardiac_phases:
            # auto seg mask for single phase has [nclasses, x, y, #slices]
            data_dict = load_data(src_data_path, cardiac_phase, type_of_data, mc_dropout=mc_dropout, dt_config_id=dt_config_id,
                                  patient_id=patient_id)
            pred_labels = data_dict["pred_labels"][patient_id]
            num_of_slices, nclasses, w, h = pred_labels.shape

            # Although we'll come here twice for each patient, data will be only added once to dataset
            if patid in train_ids:
                # training set
                is_train = True
                if patient_id not in dataset.train_patient_ids:
                    dataset.train_patient_ids.append(patient_id)
                    dataset.train_patient_num_of_slices[patient_id] = int(num_of_slices)
                    dataset.trans_dict[patient_id] = tuple((is_train, []))
                # print("Train --- patient id/image_num {}".format(patient_id))
            else:
                is_train = False
                if patient_id not in dataset.test_patient_ids:
                    dataset.test_patient_ids.append(patient_id)
                    # Store the number of slices (remember this is slices per phase) for each patient ID
                    dataset.test_patient_num_of_slices[patient_id] = int(num_of_slices)
                    dataset.trans_dict[patient_id] = tuple((is_train, []))
            # target_rois (our binary voxel labels)
            dt_labels = data_dict["dt_labels"][patient_id]
            # uncertainty maps: [w, h, #slices]
            u_maps = data_dict["umaps"][patient_id]['umap']
            # threshold all uncertainties (noise?) below 0.1 in interval [0, 1]
            # u_maps[u_maps < 0.01] = 0
            img = ACDCImage(patid, root_dir=dta_settings.short_axis_dir, resample=False,
                            scale_intensities=True)
            mri_image, _, _, frame_id = img.get(cardiac_phase=cardiac_phase)
            dataset.img_spacings[patient_id] = img.spacing
            # input_labels = RegionDetectorDataSet.convert_to_multilabel(pred_labels, labels_float)
            # TODO: re-adjust and enable previous line
            # we assume pred_labels has shape [z, nclasses, y, x]
            input_labels = create_label_contours(np.argmax(pred_labels, axis=1))
            # the target rois distinguish between the different tissue classes. We don't need this, we're only
            # interested in the ROIs in general, hence, we convert the ROI to a binary mask
            dt_labels = RegionDetectorDataSet.collapse_roi_maps(dt_labels)
            # RegionDetector object assumes all images have shape [x, y, z] (due to previous attempts), hence, in order
            # not to rewrite the whole stuff we transpose here
            mri_image, dt_labels = np.transpose(mri_image, (1, 2, 0)), np.transpose(dt_labels, (1, 2, 0))
            u_maps, input_labels = np.transpose(u_maps, (1, 2, 0)), np.transpose(input_labels, (1, 2, 0))
            dataset.augment_data(mri_image, u_maps, input_labels, dt_labels, do_rotate=is_train, is_train=is_train,
                                 patient_id=patient_id, cardiac_phase=cardiac_phase, frame_id=frame_id,
                                 pred_labels_vol=input_labels)
        # dataset.trans_dict[patient_id] contains a tuple per patient: 1) is_train 2) list with slice indices
        # print("Patient ", patient_id, dataset.trans_dict[patient_id][1])

    dataset.size_train = len(dataset.train_images)
    dataset.size_test = len(dataset.test_images)
    print(("#slices in train/test set {}/{}".format(dataset.size_train, dataset.size_test)))

    # clean up all exper handlers
    del data_dict

    return dataset


if __name__ == '__main__':
    import os
    fold = 0
    dt_config_id = "fixed_42_28"
    src_data_path = os.path.expanduser("~/expers/acdc/dcnn_mc_brierv2")
    dataset = create_dataset(fold, src_data_path, mc_dropout=False, num_of_input_chnls=3, limited_load=False,
                             dt_config_id=dt_config_id, cardiac_phases=tuple(('ES', 'ED')))
