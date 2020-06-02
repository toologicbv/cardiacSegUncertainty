import models.dcnn_mc.dilated_cnn
import models.region_detector.region_detector
import models.region_detector.region_deconv_detector
import models.region_detector.dense_net
import torch
import shutil
import dill
import os
import torch.nn as nn
import argparse

from config.region_detector.general_setup import config_detector
import utils.region_detector.batch_handler


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


class ModelHandler(object):

    def __init__(self, run_args, app_config, logger=None,
                 verbose=False, type_of_map=None, use_3d_input=False):
        """

        :param model_name:
        :param fold_id:
        :param app_config:
        :param use_cuda:
        :param use_3d_input: boolean, use 3 slabs as input (RegionDetector)
        :param str_loss_function: Used for dcnn_acdc models (brier, cross_entropy, softdice)
        :param logger:
        :param verbose:
        :param type_of_map: Used for rd models (u_map or e_map)
        :param num_io_chnnls: number of channels in the input layer. One model can process different channels numbers
        """
        if isinstance(run_args, argparse.Namespace):
            run_args = vars(run_args)
        self.verbose = verbose
        self.use_cuda = run_args["cuda"]
        self.model_name = run_args["model"]
        self.logger = logger
        self.use_3d_input = use_3d_input
        self.fold_id = run_args["fold_id"]
        self.type_of_map = type_of_map
        self.app_config = app_config
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.param_file_name = None
        self.checkpoint = None
        self.model = None
        self.model_package = None
        self.model_architecture = app_config.get_architecture(model_name=self.model_name,
                                                              num_input_channels=None,
                                                              do_return=True)
        if self.model_architecture is None:
            self.model_architecture = app_config.architecture
        self._set_model_dependent_attributes()

        # in case we load a model from file AND we saved the config object, we can load the config file and
        # attach if to the model handler. Can be useful after training to inspect the details of the configuration
        # used during training (bookkeeping). See load_parameters method
        self.model_app_config = None
        self._set_model_param_dir()

    def _set_model_dependent_attributes(self):
        self.batch_class = getattr(utils.region_detector.batch_handler, self.app_config.batch_handler_class)
        self.num_input_channels = self.model_architecture["num_of_input_channels"]

        if self.model_name[:2] == "rd" and self.model_name != "rd4":
            self.str_model_class = "RegionDetector"
            self.model_package = models.region_detector.region_detector
            if self.model_name == "rd2":
                self.learning_label_type = "grid4"
            else:
                self.learning_label_type = None

        elif self.model_name[:3] == "rd4":
            self.str_model_class = "RegionDetectorWithDeconv"
            self.model_package = models.region_detector.region_deconv_detector
            self.learning_label_type = "deconv"

        elif self.model_name[:5] == "dsnet":
            self.str_model_class = "DenseNet"
            self.model_package = models.region_detector.dense_net
            self.learning_label_type = None
        else:
            raise ValueError

    def _set_model_param_dir(self):
        subdir = os.path.join("region_detector", self.model_name)
        self.model_param_dir = os.path.join(self.app_config.model_root_dir, subdir)
        self.model_param_dir = os.path.join(self.model_param_dir, "parameters")
        self.model_param_dir = os.path.join(self.model_param_dir, "fold" + str(self.fold_id))
        if self.type_of_map is not None:
            self.model_param_dir = os.path.join(self.model_param_dir, self.type_of_map)

    def _create_default_instance(self):

        if self.model_name[:2] == 'rd' or self.model_name[:5] == "dsnet":
            model_class = getattr(self.model_package, self.str_model_class)
            message = "Creating new model {}: {}".format(self.str_model_class, self.model_architecture["description"])
            self.info(message)
            if self.verbose:
                self.info("Network configuration details")
                for c_key, c_value in self.model_architecture.items():
                    c_msg = "{} = {}".format(c_key, str(c_value))
                    self.info(c_msg)
                config_det_msg = "#max-pool = {}".format(self.app_config.num_of_max_pool)
                self.info(config_det_msg)
                config_det_msg = "fraction_negatives = {}".format(self.app_config.fraction_negatives)
                self.info(config_det_msg)
                config_det_msg = "max_grid_spacing = {}".format(self.app_config.max_grid_spacing)
                self.info(config_det_msg)

            # learning rate is just a default
            model = model_class(self.model_architecture, lr=0.0001, use_3d_input=self.use_3d_input)
        else:
            raise ValueError("{} name is unknown and hence cannot be created".format(self.model_name))

        self.model = model
        # assign model to CPU or GPU if available
        self.model = self.model.to(self.device)

    def initialize(self, lr):
        if self.model is None:
            self._create_default_instance()
        self.model.set_learning_rate(lr)

    def get_model(self):
        if self.model is None:
            self._create_default_instance()
        return self.model

    def load_parameters(self, checkpoint=150000, subdir_path=None):
        self.checkpoint = checkpoint
        if self.model_name[:2] == 'rd':
            checkpoint_file = self.str_model_class + "checkpoint" + str(self.checkpoint).zfill(5) + ".pth.tar"
            config_checkpoint_file = checkpoint_file.replace(".tar", "_config.dll")
            self._create_default_instance()
        if subdir_path is not None:
            self.model_param_dir = os.path.join(self.model_param_dir, subdir_path)
        self.param_file_name = os.path.join(self.model_param_dir, checkpoint_file)
        self.info("INFO - load from dir {}".format(self.param_file_name))
        if os.path.exists(self.param_file_name):
            model_state_dict = torch.load(self.param_file_name)
            self.model.load_state_dict(model_state_dict["state_dict"])
            if self.use_cuda:
                self.model.cuda()
            self.info("Loading existing model with checkpoint {} from dir {}".format(checkpoint, self.param_file_name))
        else:
            self.info("Path to checkpoint not found {}".format(self.param_file_name))
            raise IOError

        try:
            config_checkpoint_file = os.path.join(self.model_param_dir, config_checkpoint_file)
            if os.path.exists(config_checkpoint_file):
                with open(config_checkpoint_file, "rb") as f:
                    self.model_app_config = dill.load(f)
                self.info("INFO - Successfully loaded config object from {}".format(config_checkpoint_file))
        except Exception as e:
            # this is an optional step, hence we don't care if there is no config file stored, just continue
            pass

    def save_checkpoint(self, state, is_best, save_root_dir=None, prefix=None, filename='checkpoint{}.pth.tar',
                        save_config=False):
        """

        :param state: dictionary that holds some detailed information about the model state
        :param is_best: boolean, actually we don't use this currently
        :param save_root_dir: If None we use the self.model_param_dir to save the checkpoint (actually during
                              training of the model we use the exper handler dir and later during evaluation
                              we can move the model parameter file to a different directory and load it form there,
                              i.e. we use the model_param_dir then.
        :param prefix: string, used as prefix for filename. Currently equal to the ObjectClass of the model
                        e.g. RegionDetector
        :param filename:
        :param save_config: boolean, if true we also save the config object. The save-filename is equal to
                            the model-save-filename but we replace ".tar" with "_config.dll".
                            Can be used after training to inspect the details about the model architecture.
        :return:
        """
        filename = filename.format(str(state["epoch"]).zfill(5))
        if save_root_dir is None:
            save_root_dir = self.model_param_dir

        if prefix is not None:
            file_name = os.path.join(save_root_dir, prefix + filename)
        else:
            file_name = os.path.join(save_root_dir, filename)

        self.info("INFO - Saving model at epoch {} to {}".format(state["epoch"], file_name))
        torch.save(state, file_name)
        if is_best:
            shutil.copyfile(file_name, file_name + '_model_best.pth.tar')
        if save_config:
            config_filename = file_name.replace(".tar", "_config.dll")
            self.app_config.save_config(config_filename)

    def info(self, message):
        if self.logger is None:
            print(message)
        else:
            self.logger.info(message)

    def _print_architecture_details(self):
        self.info("Network configuration details")
        for c_key, c_value in self.model_architecture.items():
            c_msg = "{} = {}".format(c_key, str(c_value))
            self.info(c_msg)


if __name__ == "__main__":
    run_args = {'cuda': True, 'model': "dsnet1", 'fold_id': 0}
    model_handler = ModelHandler(run_args, app_config=config_detector, use_3d_input=False,)
    model_handler.initialize(lr=0.0001)
    # model_handler.load_parameters(checkpoint=60000)

