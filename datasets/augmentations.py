import numpy as np
import torch
from torch.nn.modules.utils import _pair
import cv2
from scipy.ndimage import gaussian_filter


def create_new_sample(image, reference, spacing, sample):
    new_sample = {'image': image,
                  'reference': reference,
                  'spacing': spacing,
                  'patient_id': sample['patient_id']}
    new_sample.update((newkey, newvalue) for newkey, newvalue in sample.items() if newkey not in new_sample.keys())
    return new_sample


class RandomMirroring(object):
    def __init__(self, axis, p=0.5, rs=np.random):
        self.axis = axis
        self.p = p
        self.rs = rs

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']
        rs = self.rs
        if rs.rand() < 0.5:
            image = np.flip(image, axis=self.axis)
            reference = np.flip(reference, axis=self.axis)

        new_sample = create_new_sample(image, reference, spacing, sample)
        del sample
        return new_sample


class RandomPerspective(object):
    def __init__(self, rs=np.random):
        self.rs = rs

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']
        M = np.identity(3, float)
        M += self.rs.uniform(-0.001, 0.001, (3, 3))
        shape = (image.shape[-1], image.shape[-2]) # works for 2D and 3D
        # print('perspective', image.shape)

        if image.ndim == 3:
            # print(image.transpose(1,2,0).shape, M.shape, shape)
            # print(cv2.warpPerspective(image.transpose(1,2,0), M, shape, flags=cv2.INTER_LINEAR).shape)
            image = cv2.warpPerspective(image.transpose(1,2,0), M, shape, flags=cv2.INTER_LINEAR).transpose(2,0,1)
        else:
            image = cv2.warpPerspective(image, M, shape, flags=cv2.INTER_LINEAR)
        reference = cv2.warpPerspective(reference, M, shape, flags=cv2.INTER_NEAREST)
        new_sample = create_new_sample(image, reference, spacing, sample)
        del sample
        return new_sample


class RandomRotation(object):
    def __init__(self, axes, rs=np.random):
        self.axes = axes
        self.rs = rs

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']
        k = self.rs.randint(0, 4)
        image = np.rot90(image, k, self.axes)
        reference = np.rot90(reference, k, self.axes)

        new_sample = create_new_sample(image, reference, spacing, sample)
        # print("INFO - RandomRotation - new_sample[image] ", new_sample['image'].shape)
        del sample
        return new_sample


class RandomIntensity(object):
    def __init__(self, rs=np.random):
        self.rs = rs
        # self.maximum_g = 1.25
        # self.maximum_gain = 10

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']



        # transform = self.rs.randint(2)
        # if transform == 0:
        #     pass
        # elif transform == 1:
        gain = self.rs.uniform(2.5, 7.5)
        cutoff = self.rs.uniform(0.25, 0.75)
        image = (1 / (1 + np.exp(gain * (cutoff - image))))
        # else:
        #     g = self.rs.rand() * 2 * self.maximum_g - self.maximum_g
        #     if g < 0:
        #         g = 1 / np.abs(g)
        #     image = image**g

        new_sample = create_new_sample(image, reference, spacing, sample)
        del sample
        return new_sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    TODO: Random sampler should be dependent on a specific random state. Same samples should be shown regardless of network used.
    """

    def __init__(self, output_size, input_padding=None, rs=np.random):
        assert isinstance(output_size, (int, tuple))
        self.rs = rs
        self.output_size = _pair(output_size)
        self.input_padding = input_padding

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']
        h, w = reference.shape
        new_h, new_w = self.output_size
        # print("ERROR ", sample['patient_id'], h, w, self.output_size, self.input_padding, new_h, new_w, (h-new_h), (w-new_w))
        rs = self.rs

        top = rs.randint(0, h - new_h)
        left = rs.randint(0, w - new_w)

        reference = reference[top:  top + new_h,
                              left: left + new_w]

        if self.input_padding:
            new_h += 2*self.input_padding
            new_w += 2*self.input_padding
        if image.ndim == 3:
            image = image[:, top:  top + new_h,
                              left: left + new_w]
        else:
            image = image[top:  top + new_h,
                          left: left + new_w]

        new_sample = create_new_sample(image, reference, spacing, sample)
        # print("INFO - RandomCrop - new_sample[image] ", new_sample['image'].shape)
        del sample
        return new_sample


class PadInput(object):

    def __init__(self, pad, output_size=None):
        self.pad = pad
        self.output_size = output_size

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']

        if self.pad > 0:
            if self.output_size is not None:
                # IMPORTANT: Make sure the shape of the original image/reference is never smaller than the new
                # output/patch size. Otherwise our random crop crashes.
                image, reference = self._check_shape(image, reference)
            if image.ndim == 3:
                # image = np.pad(image, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='edge')
                image = np.pad(image, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                               'constant', constant_values=(0,))
            else:
                # image = np.pad(image, ((self.pad, self.pad), (self.pad, self.pad)), mode='edge')
                image = np.pad(image, ((self.pad, self.pad), (self.pad, self.pad)), 'constant', constant_values=(0,))

        new_sample = create_new_sample(image, reference, spacing, sample)
        # print("INFO - PadInput - new_sample[image] ", new_sample['image'].shape)
        del sample
        return new_sample

    def _check_shape(self, image, reference):

        h, w = reference.shape
        new_h, new_w = self.output_size
        pad_h, pad_w = 0, 0
        if (h - new_h) <= 0:
            pad_h = abs(h - new_h) + 1
            pad_h = pad_h/2 if pad_h % 2 == 0 else (pad_h + 1) / 2
        if (w - new_w) <= 0:
            pad_w = abs(w - new_w) + 1
            pad_w = pad_w / 2 if pad_w % 2 == 0 else (pad_w + 1) / 2

        if pad_w != 0 or pad_h != 0:
            # If necessary we pad to at least the size of patch size = output size
            pad_h, pad_w = int(pad_h), int(pad_w)
            if image.ndim == 3:
                image = np.pad(image, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='edge')
                reference = np.pad(reference, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            else:
                image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
                reference = np.pad(reference, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

        return image, reference


class BlurImage(object):
    def __init__(self, sigma=1., rs=None):
        self.sigma = sigma
        self.rs = rs

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32)

        if image.ndim == 3:
            new_image = np.zeros_like(image)
            for i in range(image.shape[0]):
                new_image[i] = gaussian_filter(image[i], self.sigma)
        elif image.ndim == 2:
            new_image = gaussian_filter(image, self.sigma)
        else:
            raise ValueError("BlurImage - Error - ndim not supported {}".format(image.ndim))
        new_sample = create_new_sample(new_image, reference, spacing, sample)
        del sample
        return new_sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, reference, spacing = sample['image'], sample['reference'], sample['spacing']

        try:
            # image = torch.from_numpy(image).float()
            # reference = torch.from_numpy(reference).long()
            image = torch.FloatTensor(image, device='cpu')
            reference = torch.LongTensor(reference, device='cpu')
        except ValueError:
            # image = torch.from_numpy(np.ascontiguousarray(image)).float()
            # reference = torch.from_numpy(np.ascontiguousarray(reference)).long()
            image = torch.FloatTensor(np.ascontiguousarray(image), device='cpu')
            reference = torch.LongTensor(np.ascontiguousarray(reference), device='cpu')

        new_sample = {'image': image[None],
                      'reference': reference,
                      'original_spacing': torch.FloatTensor(sample['original_spacing'], device='cpu'),
                      'spacing': torch.FloatTensor(spacing, device='cpu'),
                      'patient_id': sample['patient_id'],
                      'cardiac_phase': sample['cardiac_phase'],
                      'frame_id': torch.IntTensor(np.array([sample['frame_id']]), device='cpu')}

        if 'ignore_label' in new_sample.keys():
            ignore_label = new_sample['ignore_label']
            if isinstance(ignore_label, np.ndarray):
                new_sample['ignore_label'] = torch.from_numpy(ignore_label).float()
        del sample

        # print("INFO - ToTensor - ", new_sample['image'].dtype)
        return new_sample

