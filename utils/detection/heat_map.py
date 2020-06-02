import numpy as np
import os
import glob

from utils.detection.batch_handler import create_grid_heat_map
from evaluate.common import save_nifty
from datasets.common import read_nifty


class ImageHeatMapHandler(object):

    def __init__(self, mode_dataset="separate", spacings=None):

        if mode_dataset == "separate" or mode_dataset[:6] == "single":
            self.phase_split = True
        else:
            self.phase_split = False
        self.heat_maps = {}
        self.spacings = spacings

    def _check_dict(self, patient_id):
        if patient_id not in self.heat_maps.keys():
            self.heat_maps[patient_id] = {}

    def add_patient_slice_pred_probs(self, patient_id, slice_id, pred_probs, frame_id, grid_spacing,
                                     patch_img_size, patch_slice_xy, org_img_size, num_of_slices,
                                     create_heat_map=True):
        """

        :param patient_id:
        :param slice_id: NOTE: runs from 1...num of slices
        :param frame_id: the actual cardiac frame id (integer for ED and ES)
        :param pred_probs:
        :param grid_spacing: currently 4 or 8 (defined in config object for experiment)
        :param patch_img_size: tuple(w,h) patch size we use during training/testing, cropped compared to original
        :param org_img_size: these are the width, height of the original data set image, SO NOT THE CROPPED ONE WE ARE WORKING
                                WITH DURING REGION DETECTION!!!
        :param patch_slice_xy: tuple(slice_x, slice_y) size of the cropped image

        :return: We convert the softmax predictions for each tiled slice back into a heat-map/pred_probs map of the original
                 image slice, in order to be able to use these to compute the effect of the overall segmentation
                 performance.
                 The BATCH property batch_patient_pred_probs is a dictionary with key=patient_id and frame_id
                 (for cardiac phase, so not ED/ES but real frame_id, an integer)
                 Each entry contains a numpy array of shape [w, h, #slices] where w and h corresponds to the original
                 image size
        """
        if pred_probs.ndim == 3:
            # get rid of first dummy batch dimension (equal to 1)
            pred_probs = np.squeeze(pred_probs)

        slice_id -= 1
        if create_heat_map:
            heat_map, _, _ = create_grid_heat_map(pred_probs, grid_spacing, w=patch_img_size[0], h=patch_img_size[1],
                                                  prob_threshold=0)
        else:
            # in case we use the deconv model we already have voxel-wise probs
            heat_map = pred_probs

        if self.phase_split:
            self._add_single_phase(patient_id, frame_id, org_img_size, num_of_slices, patch_slice_xy, slice_id,
                                   heat_map)
        else:
            self._add_combined_phase(patient_id, frame_id, org_img_size, num_of_slices, patch_slice_xy, slice_id,
                                     heat_map)

    def _add_single_phase(self, patient_id, frame_id, org_img_size, num_of_slices, patch_slice_xy, slice_id,
                          heat_map):
        if patient_id not in list(self.heat_maps.keys()) or \
                (patient_id in list(self.heat_maps.keys()) and
                 frame_id not in list(self.heat_maps[patient_id].keys())):
            if patient_id not in list(self.heat_maps.keys()):
                self.heat_maps[patient_id] = {}
            # initialize numpy array
            p = np.zeros((num_of_slices, org_img_size[0], org_img_size[1]))
            p[slice_id, patch_slice_xy[0], patch_slice_xy[1]] = heat_map
            self.heat_maps[patient_id][frame_id] = p
        else:
            # patient as key already exists
            self.heat_maps[patient_id][frame_id][slice_id, patch_slice_xy[0], patch_slice_xy[1]] = heat_map

    def _add_combined_phase(self, patient_id, phase_id, org_img_size, num_of_slices, patch_slice_xy, slice_id,
                            heat_map):
        if patient_id not in list(self.heat_maps.keys()):
            # initialize numpy array
            p = np.zeros((2, org_img_size[0], org_img_size[1], num_of_slices))
            p[phase_id, patch_slice_xy[0], patch_slice_xy[1], slice_id] = heat_map
            self.heat_maps[patient_id] = p
        else:
            # patient as key already exists
            self.heat_maps[patient_id][phase_id, patch_slice_xy[0], patch_slice_xy[1], slice_id] = heat_map

    def save_heat_maps(self, output_dir, patient_id=None, mc_dropout=False):

        if patient_id is None:
            pat_list = list(self.heat_maps.keys())
        else:
            pat_list = [patient_id]
        if mc_dropout:
            file_suffix = "_mc.nii.gz"
        else:
            file_suffix = ".nii.gz"

        for p_id in pat_list:
            if self.phase_split:
                self._save_separate_maps(p_id, output_dir, file_suffix)
            else:
                self._save_combined_maps(p_id, output_dir, file_suffix)
        print(("INFO - Successfully saved {} files to {}".format(len(pat_list), output_dir)))

    def load_heat_map(self, input_dir, patient_id, frame_id, mc_dropout=False):
        if mc_dropout:
            file_suffix = "_mc.nii.gz"
        else:
            file_suffix = ".nii.gz"

        if self.phase_split:
            self._load_separate_map(input_dir, patient_id, frame_id, file_suffix)
        else:
            raise NotImplementedError

    def _save_separate_maps(self, patient_id, output_dir, file_suffix):

        for frame_id in self.heat_maps[patient_id].keys():
            if isinstance(frame_id, str):
                fname_suffix = "_{}".format(frame_id) + file_suffix
            else:
                fname_suffix = "_{:02d}".format(frame_id) + file_suffix
            file_name_out = patient_id + fname_suffix
            file_name_out = os.path.join(output_dir, file_name_out)
            heat_map = self.heat_maps[patient_id][frame_id]
            save_nifty(heat_map, self.spacings[patient_id], file_name_out, direction=None, origin=None)

    def _save_combined_maps(self, patient_id, output_dir, file_suffix):

        file_name_out = os.path.join(output_dir, patient_id + file_suffix)
        heat_map = self.heat_maps[patient_id]
        save_nifty(heat_map, self.spacings[patient_id], file_name_out, direction=None, origin=None)

    def get_heat_map(self, patient_id, frame_id=None):
        if frame_id is not None:
            return self.heat_maps[patient_id][frame_id]
        else:
            return self.heat_maps[patient_id]

    def _load_separate_map(self, input_dir, patient_id, frame_id, file_suffix):
        fname_suffix = "_{}".format(frame_id) + file_suffix
        file_name_in = patient_id + fname_suffix
        file_name_in = os.path.join(input_dir, file_name_in)
        self._check_dict(patient_id)
        self.heat_maps[patient_id][frame_id], spacing = read_nifty(file_name_in, get_extra_info=False)

    @property
    def patient_ids(self):
        return list(self.heat_maps.keys())

    def load(self, input_dir, cardiac_phase, verbose=False, mc_dropout=False):
        if mc_dropout:
            file_suffix = "_{}_mc.nii.gz".format(cardiac_phase)
        else:
            file_suffix = "_{}.nii.gz".format(cardiac_phase)
        search_mask = os.path.join(input_dir, "patient*" + file_suffix)
        file_list = glob.glob(search_mask)
        if len(file_list) == 0:
            raise ValueError("ERROR - Can't find any heat maps with search mask {}".format(search_mask))
        file_list.sort()
        files_loaded = 0
        for fname in file_list:
            # assuming fname is e.g. /home/jorg/.../patient037_ES.npz
            base_filename = os.path.splitext(os.path.basename(fname))[0]
            # remove file extension
            base_filename = base_filename.strip(".nii.gz")
            f_parts = base_filename.split("_")
            patient_id, cardiac_phase = f_parts[0], f_parts[1]
            self._check_dict(patient_id)
            self.heat_maps[patient_id][cardiac_phase], spacing = read_nifty(fname)
            files_loaded += 1
        if verbose:
            print("INFO - Loaded {} heat maps".format(files_loaded))


if __name__ == "__main__":
    heat_map_handler = ImageHeatMapHandler()
    heat_map_handler.load(input_dir="/home/jorg/models/region_detector/rd1/heat_maps/e_map/")
    print(heat_map_handler.heat_maps.keys())
