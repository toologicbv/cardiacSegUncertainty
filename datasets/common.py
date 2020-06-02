import os
import yaml
import glob
import scipy.ndimage
import numpy as np
import SimpleITK as sitk

from datasets.data_config import get_config


def get_image_file_list(search_mask_img) -> list:
    files_to_load = glob.glob(search_mask_img)
    files_to_load.sort()
    if len(files_to_load) == 0:
        raise ValueError("ERROR - get_image_file_list - Can't find any files to load in {}".format(search_mask_img))
    return files_to_load


def apply_2d_zoom_3d(arr3d, spacing, new_spacing, order=1, do_blur=True, as_type=np.float32):
    """

    :param arr3d: [#slices, IH, IW]
    :param spacing: spacing has shape [#slices, IH, IW]
    :param new_vox_size: tuple(x, y)
    :param order: of B-spline
    :param do_blur: boolean (see below)
    :param as_type: return type of np array. We use this to indicate binary/integer labels which we'll round off
                    to prevent artifacts
    :return:
    """
    if len(spacing) > 2:
        spacing = spacing[int(len(spacing) - 2):]

    if len(new_spacing) > 2:
        new_spacing = new_spacing[int(len(new_spacing) - 2):]

    zoom = np.array(spacing, np.float64) / new_spacing
    if do_blur:
        for z in range(arr3d.shape[0]):
            sigma = .25 / zoom
            arr3d[z, :, :] = scipy.ndimage.gaussian_filter(arr3d[z, :, :], sigma)

    resized_img = scipy.ndimage.interpolation.zoom(arr3d, tuple((1,)) + tuple(zoom), order=order)
    if as_type == np.int:
        # binary/integer labels
        resized_img = np.round(resized_img).astype(as_type)
    return resized_img


def read_nifty(fname, get_extra_info=False):
    img = sitk.ReadImage(fname)
    spacing = img.GetSpacing()[::-1]
    arr = sitk.GetArrayFromImage(img)
    if get_extra_info:
        return arr, spacing, img.GetDirection(), img.GetOrigin()
    return arr, spacing


def extract_ids(file_list, f_suffix=".nii.gz"):
    new_list = []
    for fname in file_list:
        b_fname = os.path.basename(fname.strip(f_suffix))
        new_list.append(b_fname)

    return new_list


def get_arvc_datasets(split=(0.50, 0.25, 0.25), rs=None, ids_only=False) -> dict:
    """
    Creates three list with absolute file names of short-axis MRI images for ARVC dataset
    training, validation and test based on the specified split percentages.

    IMPORTANT: we first check whether we already created a certain split (split file name exists)
                if true, we load the existing file else we create a new one in data root dir e.g. ~/data/ARVC/

    :param split:
    :param rs:
    :param ids_only
    :return:
    """

    def create_absolute_file_names(rel_file_list, src_path) -> list:
        return [os.path.join(src_path, rel_fname) for rel_fname in rel_file_list]

    def get_dataset_files(all_files, file_ids) -> list:
        return [all_files[fid] for fid in file_ids]

    dta_settings = get_config('ARVC')
    if os.path.isfile(dta_settings.split_file):
        # load existing splits
        with open(dta_settings.split_file, 'r') as fp:
            split_config = yaml.load(fp, Loader=yaml.FullLoader)
            training_ids = split_config['training']
            validation_ids = split_config['validation']
            test_ids = split_config['test']
        print("INFO - Load existing split file {}".format(dta_settings.split_file))
    else:
        # create new split
        assert sum(split) == 1.
        # get a list with the short-axis image files that we have in total (e.g. in ~/data/ARVC/images/*.nii.gz)
        search_suffix = "*" + dta_settings.img_file_ext
        search_mask_img = os.path.expanduser(os.path.join(dta_settings.short_axis_dir, search_suffix))
        # we make a list of relative file names (root data dir is omitted)
        files_to_load = [os.path.basename(abs_fname) for abs_fname in get_image_file_list(search_mask_img)]
        num_of_patients = len(files_to_load)
        # permute the list of all files, we will separate the permuted list into train, validation and test sets
        if rs is None:
            rs = np.random.RandomState(78346)
        ids = rs.permutation(num_of_patients)
        # create three lists of files
        training_ids = get_dataset_files(files_to_load, ids[:int(split[0] * num_of_patients)])
        c_size = int(len(training_ids))
        validation_ids = get_dataset_files(files_to_load, ids[c_size:c_size + int(split[1] * num_of_patients)])
        c_size += len(validation_ids)
        test_ids = get_dataset_files(files_to_load, ids[c_size:])

        # write split configuration
        split_config = {'training': training_ids, 'validation': validation_ids, 'test': test_ids}
        print("INFO - Write split file {}".format(dta_settings.split_file))
        with open(dta_settings.split_file, 'w') as fp:
            yaml.dump(split_config, fp)

    return {'training': create_absolute_file_names(training_ids, dta_settings.short_axis_dir),
            'validation': create_absolute_file_names(validation_ids, dta_settings.short_axis_dir)[:25],
            'test': create_absolute_file_names(test_ids, dta_settings.short_axis_dir)}


if __name__ == "__main__":
    arvc_datasets = get_arvc_datasets(split=(0.50, 0.25, 0.25))
    print(arvc_datasets['training'])
    print(arvc_datasets['validation'])
    print(arvc_datasets['test'])
