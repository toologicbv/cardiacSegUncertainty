import os
import socket


def get_rootpath():
    return os.path.expanduser(os.environ.get("REPO_PIPELINE", "~/repository/seg_eval_pipeline"))


class Settings4d(object):

    def __init__(self):

        self.root_path = get_rootpath()
        if socket.gethostname() == "qiaubuntu" or socket.gethostname() == "toologic-ubuntu2":
            self.data_path = os.path.expanduser("~/repository/data/ACDC/all_cardiac_phases/")
        else:
            self.data_path = os.path.expanduser("~/data/ACDC/all_cardiac_phases/")
        self.max_number_of_patients = 2
        # ACDC data paths
        self.pred_labels_path_4d = self.data_path.replace('all_cardiac_phases', 'pred_labels')
        self.voxel_spacing = (1.4, 1.4)
        self.int_percentiles = (5, 99)
        # padding to left and right of the image in order to reach the final image size for classification
        self.pad_size = 65
        self.batch_batch_shape = 150
        self.batch_batch_shape_wpadding = 281
        # class labels
        self.num_of_tissue_classes = int(4)
        self.class_lbl_background = int(0)
        self.class_lbl_RV = int(1)
        self.class_lbl_myo = int(2)
        self.class_lbl_LV = int(3)


class Settings3d(object):

    def __init__(self):
        self.root_path = get_rootpath()
        if socket.gethostname() == "qiaubuntu" or socket.gethostname() == "toologic-ubuntu2":
            self.data_path = os.path.expanduser("~/repository/data/ACDC/all_cardiac_phases/")
        else:
            self.data_path = os.path.expanduser("~/data/ACDC/all_cardiac_phases/")
        self.acdc_train_path = "train"
        self.acdc_val_path = "validate"
        self.acdc_image_path = "images_resize"
        self.acdc_label_path = "reference_resize"
        self.max_number_of_patients = 2
        self.voxel_spacing = (1.4, 1.4)
        self.int_percentiles = (5, 99)
        # padding to left and right of the image in order to reach the final image size for classification
        self.pad_size = 65
        self.batch_batch_shape = 150
        self.batch_batch_shape_wpadding = 281
        # class labels
        self.num_of_tissue_classes = int(4)
        self.class_lbl_background = int(0)
        self.class_lbl_RV = int(1)
        self.class_lbl_myo = int(2)
        self.class_lbl_LV = int(3)


acdc_settings3d = Settings3d()
acdc_settings4d = Settings4d()
