import os
import glob
import numpy as np
import torch
from collections import OrderedDict
import yaml

from networks.detection.general_setup import config_detector
from utils.common import loadExperimentSettings
from utils.detection.batch_handler import BatchHandler
from utils.detection.heat_map import ImageHeatMapHandler
from utils.detection_metrics import compute_eval_metrics
from utils.detection.trainers import get_trainer


EXPER_DIRS_OLD = \
    {"dcnn_mc_dice": {"emap": ["f0_rsn_emap_08_0085_0813_105906", "f1_rsn_emap_08_0085_0813_124710",
                               "f2_rsn_emap_08_0085_0813_124852", "f3_rsn_emap_08_0085_0813_125020"],
                      "bmap": ["f0_rsn_mc_bmap_055_0085_0816_191326", "f1_rsn_mc_bmap_055_0085_0816_191259",
                               "f2_rsn_mc_bmap_055_0085_0816_191313", "f3_rsn_mc_bmap_055_0085_0816_191246"]},
     "dcnn_mc_brier": {"emap": ["f0_rsn_emap_04_0085_0816_123020", "f1_rsn_emap_04_0085_0816_123004",
                                "f2_rsn_emap_04_0085_0816_123010", "f3_rsn_emap_04_0085_0816_122957"],
                      "bmap": ["f0_rsn_mc_bmap_055_0085_0816_183320", "f1_rsn_mc_bmap_055_0085_0816_183416",
                               "f2_rsn_mc_bmap_055_0085_0816_183357", "f3_rsn_mc_bmap_055_0085_0816_183423"]},

     "drn_mc_ce": {"emap": ["f0_rsn_emap_055_0085_0816_092909", "f1_rsn_emap_055_0085_0816_085131",
                            "f2_rsn_emap_055_0085_0816_090457", "f3_rsn_emap_055_0085_0816_092213"],
                    "bmap": ["f0_rsn_mc_bmap_055_0085_0816_085229", "f1_rsn_mc_bmap_055_0085_0816_091530",
                             "f2_rsn_mc_bmap_055_0085_0816_092323", "f3_rsn_mc_bmap_055_0085_0816_085215"]
                   },
    "drn_mc_dice": {"emap": ["f0_rsn_emap_08_0085_0814_180202", "f1_rsn_emap_08_0085_0815_091059",
                              "f2_rsn_emap_08_0085_0815_091230", "f3_rsn_emap_08_0085_0815_091251"],
                     "bmap": ["f0_rsn_mc_bmap_055_0085_0821_152630", "f1_rsn_mc_bmap_055_0085_0821_152455",
                              "f2_rsn_mc_bmap_055_0085_0821_152519", "f3_rsn_mc_bmap_055_0085_0821_152419"]},
    "unet_mc_dice": {"emap": ["f0_rsn_emap_08_0085_0812_180141", "f1_rsn_emap_08_0085_0813_085620",
                               "f2_rsn_emap_08_0085_0813_102851", "f3_rsn_emap_08_0085_0813_121427"],
                      "bmap": ["f0_rsn_mc_bmap_055_0085_0817_104717", "f1_rsn_mc_bmap_055_0085_0817_104742",
                               "f2_rsn_mc_bmap_055_0085_0817_104731", "f3_rsn_mc_bmap_055_0085_0817_104752"]},
    "unet_mc_ce": {"emap": ["f0_rsn_emap_05_0085_0816_112426", "f1_rsn_emap_05_0085_0816_112132",
                             "f2_rsn_emap_05_0085_0816_112155", "f3_rsn_emap_05_0085_0816_111410"],
                    "bmap": ["f0_rsn_mc_bmap_055_0085_0817_112406", "f1_rsn_mc_bmap_055_0085_0817_112341",
                             "f2_rsn_mc_bmap_055_0085_0817_112354", "f3_rsn_mc_bmap_055_0085_0817_112329"]},
     }


def load_settings(fname):
    with open(fname, 'r') as fp:
        args = yaml.load(fp, Loader=yaml.FullLoader)
    return args


class EvaluationHandler(object):

    def __init__(self, path_model_dir, config=config_detector, eval_run_id=None, checkpoint=None):

        self.eval_run_id = eval_run_id
        self.config = config
        self.test_set = None
        self.checkpoint = checkpoint
        experiment_settings = loadExperimentSettings(os.path.join(path_model_dir, 'settings.yaml'))
        self.architecture = load_settings(os.path.join(path_model_dir, "architecture.yaml"))
        str_checkpoint = str(checkpoint) if checkpoint is not None else str(experiment_settings.max_iters)
        model_file = os.path.join(path_model_dir, str_checkpoint + '.model')
        print(model_file)
        self._get_model_settings(experiment_settings)
        self.trainer = get_trainer(experiment_settings, self.architecture, model_file=model_file)
        self.model_tag = self._create_model_tag(experiment_settings, self.architecture)
        if self.mc_dropout:
            self.type_of_map = "bmap"
        else:
            self.type_of_map = "emap"
        # init
        self.mean_tpos_rate = []
        self.mean_prec_rate = []
        self.detection_rate = []
        self.aucs_roc = []
        self.aucs_pr = []
        self.eval_loss = []
        self.arr_eval_metrics = []  # store f1, roc_auc, pr_auc, precision, recall scores
        self.num_of_eval_slices = 0
        self.num_of_pos_eval_slices = 0
        self.num_of_neg_eval_slices = 0
        self.stats_auc_roc = None
        self.last_test_id = 0
        self.val_stats = None
        self.test_stats = None

        self.abs_root_dir = path_model_dir
        self.output_dir_results = None
        self.figure_dir = None
        self.heat_map_dir = None
        self.heat_map_handler = None
        self.heat_maps = OrderedDict()
        self._check_dirs()
        self.batch_object = None
        self.results = {}

    def _get_model_settings(self, experiment_settings):
        self.fold = experiment_settings.fold
        if "input_channels" not in vars(experiment_settings).keys():
            experiment_settings.input_channels = "allchannels"
        self.input_channels = experiment_settings.input_channels
        self.dt_config_id, self.mc_dropout = experiment_settings.dt_config_id, experiment_settings.mc_dropout

    def _create_model_tag(self, settings, architecture):
        return settings.network + "_" + str(architecture['fn_penalty_weight']) + "_" + str(architecture['fp_penalty_weight'])

    def next_test_id(self):
        self.last_test_id += 1
        return self.last_test_id

    def _check_dirs(self):
        if self.output_dir_results is None:
            self.output_dir_results = os.path.join(self.abs_root_dir, "results")
            if self.eval_run_id is not None:
                self.output_dir_results = os.path.join(self.output_dir_results, self.eval_run_id)

        if not os.path.isdir(self.output_dir_results):
            os.makedirs(self.output_dir_results, exist_ok=False)

        if self.heat_map_dir is None:
            self.heat_map_dir = os.path.join(self.abs_root_dir, "heat_maps")
            if self.eval_run_id is not None:
                self.heat_map_dir = os.path.join(self.heat_map_dir, self.eval_run_id)

        if not os.path.isdir(self.heat_map_dir):
            os.makedirs(self.heat_map_dir, exist_ok=False)

        if self.figure_dir is None:
            self.figure_dir = os.path.join(self.abs_root_dir, "figures")
            if self.eval_run_id is not None:
                self.figure_dir = os.path.join(self.figure_dir, self.eval_run_id)

        if not os.path.isdir(self.figure_dir):
            os.makedirs(self.figure_dir, exist_ok=False)

    def eval(self, data_set, batch_size=None, keep_batch=False, disrupt_chnnl=None,
             save_results=False):
        if batch_size is None:
            batch_size = data_set.get_size(is_train=False)
        self.heat_map_handler = ImageHeatMapHandler("separate", spacings=data_set.img_spacings)
        batch = BatchHandler(data_set=data_set, is_train=False, verbose=False,
                                 keep_bounding_boxes=False, input_channels=self.input_channels,
                                 num_of_max_pool_layers=self.architecture['num_of_max_pool'],
                                 app_config=self.config)
        self.trainer.evaluate(batch, keep_batch=True, batch_size=batch_size,
                              heat_map_handler=self.heat_map_handler)
        print("Evaluation #slices={}(skipped {}) (negatives/positives={}/{}) loss {:.3f}: "
              "pr_auc={:.3f} - prec={:.3f} "
              "- rec={:.3f} ".format(batch_size,
                                            0 if not keep_batch else batch.num_of_skipped_slices,
                                            self.trainer.validation_metrics['tn_slice'],
                                                     self.trainer.validation_metrics['tp_slice'],
                                                     self.trainer.current_validation_loss,
                                                     self.trainer.validation_metrics['pr_auc'],
                                                     self.trainer.validation_metrics['prec'],
                                                     self.trainer.validation_metrics['rec']))
        dt_slice_tp = self.trainer.validation_metrics['tp_slice'] / (self.trainer.validation_metrics['tp_slice'] + \
                                                                     self.trainer.validation_metrics['fn_slice'])
        dt_slice_tn = self.trainer.validation_metrics['tn_slice'] / (self.trainer.validation_metrics['tn_slice'] + \
                                                                     self.trainer.validation_metrics['fp_slice'])
        print("Evaluation - Slice detection rate tp-rate{:.3f} tn-rate {:.3f}".format(dt_slice_tp, dt_slice_tn))
        dt_rate = self.trainer.validation_metrics['detected_voxel_count'] / self.trainer.validation_metrics['total_voxel_count']
        print("Evaluation - Voxel detection rate {:.3f}".format(dt_rate))
        if keep_batch:
            self.batch_object = batch
        else:
            del batch
        if save_results:
            self.save_results()
            self.save_heat_maps()

    def save_results(self):
        if self.mc_dropout:
            file_suffix = "_mc.npz"
        else:
            file_suffix = ".npz"
        for patient_id, frame_ids in self.batch_object.batch_patient_pred_probs.items():
            for frame_id, pred_probs in frame_ids.items():
                if isinstance(frame_id, str):
                    fname_suffix = "_{}_pred_probs".format(frame_id) + file_suffix
                else:
                    fname_suffix = "_frame{:02d}_pred_probs".format(frame_id) + file_suffix
                gt_labels = self.batch_object.batch_patient_gt_labels[patient_id][frame_id]
                gt_voxel_count = self.batch_object.batch_patient_gt_label_counts[patient_id][frame_id]
                fname = patient_id + fname_suffix
                fname = os.path.join(self.output_dir_results, fname)
                np.savez(fname, **pred_probs)
                fname = fname.replace("pred_probs", "gt_labels")
                np.savez(fname, **gt_labels)
                fname = fname.replace("gt_labels", "gt_voxel_count")
                np.savez(fname, **gt_voxel_count)
        print("INFO - Saved pred_probs, gt_labels, gt_voxel_count to {}".format(self.output_dir_results))

    def save_heat_maps(self, patient_id=None):
        self._check_dirs()
        self.heat_map_handler.save_heat_maps(output_dir=self.heat_map_dir, patient_id=patient_id,
                                             mc_dropout=self.mc_dropout)

    def _check_dict(self, mydict, patient_id):
        if patient_id not in mydict.keys():
            mydict[patient_id] = {}

    def load_model(self, fname):
        state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict['model_dict'])
        print("INFO - Loaded model from {}".format(fname))


def move_results_to_root_dir(root_dir_to, input_channel, exper_root_dir):
    import shutil
    src_heat_map_dir = os.path.join(exper_root_dir, "heat_maps" + os.sep + "*.nii.gz")
    heat_maps = glob.glob(src_heat_map_dir)
    src_dt_result_dir = os.path.join(exper_root_dir, "results" + os.sep + "*.npz")
    sim_results = glob.glob(src_dt_result_dir)

    heat_map_dir = os.path.join(root_dir_to, "heat_maps")
    if not os.path.isdir(heat_map_dir):
        os.makedirs(heat_map_dir)
    dt_results_dir = os.path.join(root_dir_to, "dt_results" + os.sep + input_channel)
    if not os.path.isdir(dt_results_dir):
        os.makedirs(dt_results_dir)
    for src_fname in heat_maps:
        src_fname_base = os.path.basename(src_fname)
        shutil.copy(src_fname, os.path.join(heat_map_dir, src_fname_base))
    print("INFO - Moved {} heat maps to {}".format(len(heat_maps), heat_map_dir))

    for src_fname in sim_results:
        src_fname_base = os.path.basename(src_fname)
        shutil.copy(src_fname, os.path.join(dt_results_dir, src_fname_base))
    print("INFO - Moved {} sim result files ".format(len(sim_results)))


def run_eval_for_list_of_expers(model_dir, save_results=False):
    from datasets.region_detection.create_detector_dataset import create_dataset

    model_settings = loadExperimentSettings(os.path.join(model_dir, 'settings.yaml'))
    architecture = loadExperimentSettings(os.path.join(model_dir, 'architecture.yaml'))

    print("WARNING - Using {} for detection labels - "
          "fold: {} dropout: {}".format(model_dir, model_settings.fold, model_settings.mc_dropout))
    dataset = create_dataset(model_settings.fold,
                             src_path_data, mc_dropout=model_settings.mc_dropout,
                             num_of_input_chnls=architecture.n_channels_input,
                             limited_load=False, dt_config_id=model_settings.dt_config_id,
                             cardiac_phases=tuple(('ES', 'ED')), test_only=True)

    eval_handler = EvaluationHandler(path_model_dir=model_dir, config=config_detector, eval_run_id=None,
                                     checkpoint=None)

    # if disrupt_chnnl=4 all to-be-detected voxels will get a high uncertainty (max value e.g. 2=entropy)
    eval_handler.eval(dataset, batch_size=None, keep_batch=True,
                      disrupt_chnnl=None, save_results=save_results)
    if eval_handler.batch_object is not None:
        eval_handler.batch_object.fill_trans_dict()

# "drn_mc_dice": {"emap": ["f0_rsn_emap_16_0085_1206_094904", "f1_rsn_emap_16_0085_1206_105817",
#                              "f2_rsn_emap_16_0085_1206_105752", "f3_rsn_emap_16_0085_1206_105723"],
# "unet_mc_ce": {"emap": ["f0_rsn_emap_22_0085_1206_111051", "f1_rsn_emap_22_0085_1206_115416",
#                              "f2_rsn_emap_22_0085_1206_115443", "f3_rsn_emap_22_0085_1206_115501"],
# "drn_mc_ce": {"emap":  ["f0_rsn_emap_14_0085_1205_141715", "f1_rsn_emap_14_0085_1205_174629",
#                              "f2_rsn_emap_14_0085_1205_174717", "f3_rsn_emap_14_0085_1205_174803"],
#                     "bmap": ["f0_rsn_mc_bmap_08_0085_1205_163230", "f1_rsn_mc_bmap_08_0085_1205_171302",
#                              "f2_rsn_mc_bmap_08_0085_1205_163254", "f3_rsn_mc_bmap_08_0085_1205_171451"]}
# "dcnn_mc_dice": {"emap": ["f0_rsn_emap_10_0085_1205_162120", "f1_rsn_emap_10_0085_1205_172609",
#                                "f2_rsn_emap_10_0085_1205_172643", "f3_rsn_emap_10_0085_1205_172808"],
# "bmap": ["f0_rsn_mc_bmap_08_0085_1206_094950", "f1_rsn_mc_bmap_08_0085_1206_101857",
#                               "f2_rsn_mc_bmap_08_0085_1206_101917", "f3_rsn_mc_bmap_08_0085_1206_101934"]},
# "bmap": ["f0_rsn_mc_bmap_14_0085_1206_104850", "f1_rsn_mc_bmap_14_0085_1206_115947",
#                                "f2_rsn_mc_bmap_14_0085_1206_120115", "f3_rsn_mc_bmap_14_0085_1206_120139"]},
# "bmap": ["f0_rsn_mc_bmap_08_0085_1206_103007", "f1_rsn_mc_bmap_08_0085_1206_113130",
#                              "f2_rsn_mc_bmap_08_0085_1206_113155", "f3_rsn_mc_bmap_08_0085_1206_113217"]},

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run evaluation of detection model for all folds')
    parser.add_argument('--seg_model', type=str, help='directory for experiment outputs')
    parser.add_argument('--loss_function', type=str, default="ce")
    parser.add_argument('--channels', type=str, choices=['bmap', 'emap', None], default=None)
    parser.add_argument('--eval_id',type=str, default=None)
    parser.add_argument('--run_eval', action='store_true')
    parser.add_argument('--move_files', action='store_true')

    args = parser.parse_args()

    EXPER_DIRS = \
    {"dcnn_mc_dice": {"emap": ["f0_rsn_emap_12_0085_1212_180024", "f1_rsn_emap_12_0085_1212_181507",
                               "f2_rsn_emap_12_0085_1212_181528", "f3_rsn_emap_12_0085_1212_181546"],
                      "bmap": ["f0_rsn_mc_bmap_12_0085_1205_153302", "f1_rsn_mc_bmap_12_0085_1205_173538",
                               "f2_rsn_mc_bmap_12_0085_1205_173842", "f3_rsn_mc_bmap_12_0085_1205_174030"]},
     "dcnn_mc_brier": {"emap": ["f0_rsn_emap_12_0085_1205_124400", "f1_rsn_emap_12_0085_1205_140211",
                                "f2_rsn_emap_12_0085_1205_131500", "f3_rsn_emap_12_0085_1205_140243"],
                      "bmap": ["f0_rsn_mc_bmap_12_0085_1205_132158", "f2_rsn_mc_bmap_12_0085_1205_142826",
                               "f1_rsn_mc_bmap_12_0085_1205_142745", "f3_rsn_mc_bmap_12_0085_1205_142905"]},
     "unet_mc_dice": {"emap": ["f0_rsn_emap_12_0085_1206_124323", "f1_rsn_emap_12_0085_1206_130747",
                               "f2_rsn_emap_12_0085_1206_130801", "f3_rsn_emap_12_0085_1206_130851"],
                      "bmap": ["f0_rsn_mc_bmap_12_0085_1212_144035", "f1_rsn_mc_bmap_12_0085_1212_144100",
                               "f2_rsn_mc_bmap_12_0085_1212_144126", "f3_rsn_mc_bmap_12_0085_1212_144153"]},
     "unet_mc_ce": {"emap": ["f0_rsn_emap_12_0085_1212_164536", "f1_rsn_emap_12_0085_1212_164556",
                             "f2_rsn_emap_12_0085_1212_164619", "f3_rsn_emap_12_0085_1212_164651"],
                    "bmap": ["f0_rsn_mc_bmap_12_0085_1212_154858", "f1_rsn_mc_bmap_12_0085_1212_154913",
                             "f2_rsn_mc_bmap_12_0085_1212_154935", "f3_rsn_mc_bmap_12_0085_1212_154953"]},
     "drn_mc_dice": {"emap": ["f0_rsn_emap_12_0085_1213_093740", "f1_rsn_emap_12_0085_1213_093723",
                              "f2_rsn_emap_12_0085_1213_093854", "f3_rsn_emap_12_0085_1213_093920"],
                     "bmap": ["f0_rsn_mc_bmap_12_0085_1212_142216", "f1_rsn_mc_bmap_12_0085_1212_142233",
                              "f2_rsn_mc_bmap_12_0085_1212_142255", "f3_rsn_mc_bmap_12_0085_1212_142333"]},
     "drn_mc_ce": {"emap":  ["f0_rsn_emap_12_0085_1213_094014", "f1_rsn_emap_12_0085_1213_094029",
                             "f2_rsn_emap_12_0085_1213_094045", "f3_rsn_emap_12_0085_1213_094059"],
                    "bmap": ["f0_rsn_mc_bmap_12_0085_1212_142705", "f1_rsn_mc_bmap_12_0085_1212_142730",
                             "f2_rsn_mc_bmap_12_0085_1212_142805", "f3_rsn_mc_bmap_12_0085_1212_142919"]}
     }
    if args.channels is None:
        input_channels = ["bmap", "emap"]
    else:
        input_channels = [args.channels]
    model_tag = args.seg_model + "_" + args.loss_function
    exper_mode = "new"   # new / old

    for io_channel in input_channels:
        if args.eval_id is None:
            exper_list = EXPER_DIRS[model_tag][io_channel]
        else:
            exper_list = [args.eval_id]
        if exper_mode == "old":
            exper_list = EXPER_DIRS_OLD[model_tag][io_channel]
            src_path_data = os.path.expanduser("~/expers/acdc/" + model_tag)
            src_exper_dir = os.path.expanduser("~/expers/acdc/" + model_tag + "/dt_logs/")

        for exper_tag in exper_list:
            fold_str = exper_tag[:2]
            if exper_mode == "new":
                root_dir = os.path.expanduser("~/expers/redo_acdc/")
                src_path_data = root_dir + fold_str + os.sep + model_tag
                src_exper_dir = os.path.expanduser(src_path_data + "/dt_logs/")
            model_dir = os.path.join(src_exper_dir, exper_tag)
            # Perform check on fold we're processing
            model_settings = loadExperimentSettings(os.path.join(model_dir, 'settings.yaml'))
            if int(fold_str[1]) != model_settings.fold:
                raise ValueError("ERROR - fold of model tag is not the same as in model settings"
                                 "{} != {}".format(fold_str[1], model_settings.fold))
            if args.run_eval:
                run_eval_for_list_of_expers(model_dir, save_results=True)
            if args.move_files:
                root_dir_to = root_dir + model_tag
                if not os.path.isdir(root_dir_to):
                    os.makedirs(root_dir_to)
                print("INFO - Moving files to {}".format(root_dir_to))
                move_results_to_root_dir(root_dir_to, io_channel, model_dir)

