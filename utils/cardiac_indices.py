import numpy as np


def compute_ejection_fraction(target_vol_es, target_vol_ed, spacings, myo_vol=None, weight=None, height=None):
    """

    :param target_vol_es: binary numpy array indicating target voxels of tissue structure at ES [w, h, #slices]
    :param target_vol_ed: binary numpy array indicating target voxels of tissue structure at ED [w, h, #slices]
    :param spacings: numpy array [x-spacing, y-spacing, z-spacing]
    :param weight: patient weight in kg
    :param height: patient height in cm
    :param myo_vol: None or numpy of shape [w, h, z]
    :return: volume at ES, volume at ED, Ejection fraction = (1 - (ESV/EDV) ) * 100
             Returns stroke volumes in milliliter
             And     Ejection fraction in percentage

    """
    # Myocardial mass at end-diastole
    # 1.04 gravity of myocardium g/cm3
    # _, _, z = target_vol_ed.shape
    # target_vol_es = target_vol_es[..., 1:z-1]
    # target_vol_ed = target_vol_ed[..., 1:z-1]
    num_of_voxels_es = np.count_nonzero(target_vol_es)
    num_of_voxels_ed = np.count_nonzero(target_vol_ed)
    # print("ED {} ES {} diff {}".format(num_of_voxels_ed, num_of_voxels_es, num_of_voxels_ed - num_of_voxels_es))
    esv = np.prod(spacings) * num_of_voxels_es * 1/1000  # convert to milliliter (from mm^3)
    edv = np.prod(spacings) * num_of_voxels_ed * 1/1000
    bsa = None
    if weight is not None and height is not None:
        # Body Surface Area = SQRT( (weight * height) / 3600 )
        bsa = np.sqrt((weight * height) / 3600)
        esv *= 1/bsa
        edv *= 1/bsa
    ef = (1. - esv/float(edv)) * 100
    if myo_vol is not None:
        # compute myocardial mass at end-diastole
        num_of_voxel_myo = np.count_nonzero(myo_vol)
        myo_mass = np.prod(spacings) * num_of_voxel_myo * 1/1000 * 1.04  # g/cm3
        if bsa is not None:
            myo_mass *= 1 / bsa
        return esv, edv, ef, myo_mass
    else:
        return esv, edv, ef


