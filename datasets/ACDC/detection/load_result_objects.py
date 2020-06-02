import os
import glob
import numpy as np


def load_results(src_path_data, input_channel):

    def check_dict(mydict, patient_id):
        if patient_id not in mydict.keys():
            mydict[patient_id] = {}

    assert input_channel in ['emap', 'bmap']

    results = {}

    if input_channel == 'bmap':
        search_mask = "*_pred_probs_mc.npz"
    else:
        search_mask = "*_pred_probs.npz"

    src_path_data = os.path.join(src_path_data, input_channel)
    search_path = os.path.join(src_path_data, search_mask)
    file_list = glob.glob(search_path)
    if len(file_list) == 0:
        raise ValueError("ERROR - No results found with search mask {}".format(search_path))
    for file_to_load in file_list:
        file_basename = os.path.splitext(os.path.basename(file_to_load))[0]
        fname_components = file_basename.split("_")
        patient_id, frame_id = fname_components[0], fname_components[1]
        frame_id = frame_id.strip("frame")
        check_dict(results, patient_id)
        data = np.load(file_to_load)
        if frame_id not in results[patient_id]:
            results[patient_id][frame_id] = {}
        data_dict = dict(data)
        results[patient_id][frame_id]["pred_probs"] = data_dict
        file_to_load = file_to_load.replace("pred_probs", "gt_labels")
        data = np.load(file_to_load)
        data_dict = dict(data)
        results[patient_id][frame_id]["gt_labels"] = data_dict
        file_to_load = file_to_load.replace("gt_labels", "gt_voxel_count")
        data = np.load(file_to_load)
        data_dict = dict(data)
        results[patient_id][frame_id]["gt_voxel_count"] = data_dict

    print(("INFO - Successfully loaded {} patients results from {}".format(len(results), src_path_data)))
    return results
