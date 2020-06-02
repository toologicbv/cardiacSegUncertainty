import numpy as np
from matplotlib.colors import ListedColormap


class SegColorMap(object):

    def __init__(self, alpha=0.8):
        my_seg_map = np.array([[0, 0, 0, 0],
                                [10, 170, 28, 0],  # green
                                [232, 167, 4, 0],  # orange
                                [43, 84, 206, 0],  # blue
                                [230, 230, 0, 0],  # yellow
                                [234., 9, 9, 0]]   # red
                             )
        self.np_cmap = (my_seg_map - np.min(my_seg_map)) / (np.max(my_seg_map) - np.min(my_seg_map))
        self.np_cmap[..., -1] = alpha
        # make sure the BACKGROUND class is completely transparant
        self.np_cmap[0, -1] = 0
        self.cmap = ListedColormap(self.np_cmap)
        self.cmap._init()
        self.cmap._lut[0, -1] = 0
        self.cmap._lut[:, -1] = np.linspace(0, alpha, len(my_seg_map) + 3)

    def convert_multi_labels(self, label_array):
        if label_array.dtype != np.int:
            label_array = label_array.astype(np.uint)

        return self.np_cmap[label_array]


def transparent_cmap(cmap, N=255, alpha=0.4):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, alpha, N+4)
    return mycmap



