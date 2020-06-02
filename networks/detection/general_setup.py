import os
import torch.nn as nn
import torch
import numpy as np
import dill


class BaseConfig(object):

    def __init__(self):

        # default data directory
        # remember to ADD env variable REPO_PATH on machine. REPO_PATH=<absolute path to repository >
        self.root_dir = BaseConfig.get_rootpath()
        self.model_path = "models"
        self.model_root_dir = os.path.expanduser("~/models")
        self.log_root_path = "logs/RD"
        self.figure_path = "figures"
        self.result_path = "results"
        self.stats_path = "stats"
        self.dt_map_dir = "dt_maps"
        self.heat_map_dir = "heat_maps"
        self.checkpoint_path = "checkpoints"
        self.logger_filename = "run_out.log"
        self.batch_handler_class = "BatchHandler"
        # ES: 1 = RV; 2 = MYO; 3 = LV; tuple(apex-base inter-observer-var, NON-apex-base inter-observer-var)
        # ED: 5 = RV; 6 = MYO; 7 = LV
        self.acdc_inter_observ_var = {1: [9.05, 14.05], 2: [5.8, 7.8], 3: [5.65, 8.3],  # ES
                                      5: [8.15, 12.35], 6: [5.25, 6.95], 7: [4.65, 5.9]}  # ED
        # range of area (target tissue structure per slice) from 5 to 95 percentile (we can compute this by means of
        # TestHandler.generate_bbox_target_roi method).
        self.label_area_num_bins = 10

        # IMPORTANT: we don't use PADDING because during batch training and testing we crop out a region that fits
        # the max-pool operations & convolutions. Due toe fully-conv NN we can process different sizes of images.
        self.acdc_pad_size = 0
        self.quick_run_size = 6

        # Settings for determining the target areas/pixels to be inspected. used in procedure
        # find_multiple_connected_rois (file box_utils.py)
        # 16-11-2018: IMPORTANT CHANGE:
        #             We filtered the target rois on a minimum connected size. In experiments we actually found
        #             that the detection works slightly better if we don't filter on connected componnents.
        #             HENCE, this is not in use anymore!
        self.min_roi_area = 2  # minimum number of 2D 4-connected component. Otherwise target pixels are discarded
        self.fraction_negatives = 0.67
        # patch size during training
        self.patch_size = tuple([80, 80])
        self.test_patch_size = tuple((112, 112))
        # the spacing of grids after the last maxPool layer.
        self.max_grid_spacing = 8
        self.architecture = None

        self.detector_cfg = {"rd1":
                                 {'model_id': "rd1",
                                  'base': [16, 'M', 16, 32, 'M', 32, 'M', 64],
                                  'n_channels_input': 3,
                                  'n_classes': 2,
                                  "fn_penalty_weight": 0.6,
                                  "fp_penalty_weight": 0.08,
                                  'use_batch_norm': True,
                                  'weight_decay': 0.01,
                                  'drop_prob': 0.5,
                                  "description": "rd1-detector"},
                             "drn":
                                  {'n_channels_input': 3,
                                   'n_classes': 2,
                                   "fn_penalty_weight": 8,
                                   "fp_penalty_weight": 1,
                                   'weight_decay': 0.01,
                                   'drop_prob': 0.5,
                                   "description": "Dilated residual network"},
                             "rsn":
                                 {'n_channels_input': 3,
                                  'n_classes': 2,
                                  "fn_penalty_weight": 1.2,
                                  "fp_penalty_weight": 0.085,
                                  'weight_decay': 0.01,
                                  'drop_prob': 0.5,
                                  "description": "Simple ResNet"}
                            }
        self.base_class = "RegionDetector"
        # plotting
        self.title_font_large = {'fontname': 'Monospace', 'size': '36', 'color': 'black', 'weight': 'normal'}
        self.title_font_medium = {'fontname': 'Monospace', 'size': '20', 'color': 'black', 'weight': 'normal'}
        self.title_font_small = {'fontname': 'Monospace', 'size': '16', 'color': 'black', 'weight': 'normal'}
        self.axis_font = {'fontname': 'Monospace', 'size': '16', 'color': 'black', 'weight': 'normal'}
        self.axis_font18 = {'fontname': 'Monospace', 'size': '18', 'color': 'black', 'weight': 'normal'}
        self.axis_font20 = {'fontname': 'Monospace', 'size': '20', 'color': 'black', 'weight': 'normal'}
        self.axis_font22 = {'fontname': 'Monospace', 'size': '22', 'color': 'black', 'weight': 'normal'}
        self.axis_font24 = {'fontname': 'Monospace', 'size': '24', 'color': 'black', 'weight': 'normal'}

    def get_architecture(self, model_name):
        if model_name == "rd1":
            self.max_grid_spacing = 8
            self.patch_size = tuple([80, 80])
            self.architecture = self.detector_cfg[model_name]
            self.architecture['num_of_max_pool'] = len([i for i, s in enumerate(self.architecture["base"]) if s == 'M'])
            # assuming patch size is quadratic
            self.architecture['output_stride'] = int(self.patch_size[0] / 2 ** self.architecture['num_of_max_pool'])
        elif model_name == "drn" or model_name == "rsn":
            self.max_grid_spacing = 8
            self.patch_size = tuple([80, 80])
            self.architecture = self.detector_cfg[model_name]
            self.architecture['num_of_max_pool'] = 3
            # assuming patch size is quadratic
            self.architecture['output_stride'] = int(self.patch_size[0] / 2 ** self.architecture['num_of_max_pool'])
        else:
            raise NotImplementedError("ERROR - {} is not a valid model name".format(model_name))

        # add some entries to architecture dict because we save that one
        self.architecture['patch_size'] = self.patch_size
        self.architecture['test_patch_size'] = self.test_patch_size
        self.architecture['max_grid_spacing'] = self.max_grid_spacing

    @staticmethod
    def get_rootpath():
        return os.path.expanduser(os.environ.get("REPO_PIPELINE", os.environ.get('HOME')))

    def save_config(self, abs_file_name):

        with open(abs_file_name, 'wb') as f:
            dill.dump(self, f)

        print(("INFO - Saved region detector configuration to {}".format(abs_file_name)))


config_detector = BaseConfig()
