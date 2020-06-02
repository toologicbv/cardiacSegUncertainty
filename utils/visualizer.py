# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 18:54:27 2016

@author: BobD
"""
import visdom
import numpy as np


def projections(im, zoom_factors=(1,1,1), return_full=True):
#    ratios = np.array(ratios)
    z_size, y_size, x_size = im.shape
    centers = (np.array(im.shape, int) + 1) // 2
#    im = im[::-1]
    
    axial = im[centers[0]]
    coronal = im[:, centers[1], :]
    sagittal = im[:, :, centers[2]]
#    
#    axial = interpolation.zoom(axial, zoom_factors[1:], order=1)
#    coronal = interpolation.zoom(coronal, (zoom_factors[0], zoom_factors[2]), order=1)
#    sagittal = interpolation.zoom(sagittal, zoom_factors[:2], order=1)
        
    if not return_full:
        ret = axial, coronal, sagittal
    else:
        full_im_shape = (axial.shape[0] + coronal.shape[0], axial.shape[1] + sagittal.shape[0])
        full_im = np.zeros(full_im_shape)
        full_im[:axial.shape[0], :axial.shape[1]] = axial
        full_im[axial.shape[0]:, :axial.shape[1]] = coronal
        full_im[:axial.shape[0], axial.shape[1]:] = sagittal.T
        ret = full_im
    return ret


class Visualizer(object):
    def __init__(self, env, port, title, labels):
        assert(type(labels) == list)
        self.numoflabels = len(labels)
        self.vis = visdom.Visdom(env=env, port=port)
        opts = dict(title=title, legend=labels, showlegend=True)
        self.vis_line = self.vis.line(np.zeros((1, self.numoflabels)), win=title, opts=opts)
        self.iteration = 0
        
    def __call__(self, *args):
        assert(len(args) == self.numoflabels)
        self.vis.line(np.array([args]),
                      np.array([[self.iteration] * len(args)]),
                      win=self.vis_line, 
                      update=('append' if self.iteration!=0 else 'replace'))
        self.iteration += 1

    def image(self, arr, title, win=2):
        self.vis.image(arr, win=win, opts=dict(title=title))

    def text(self, s, title, win=3):
        self.vis.text(s, win=win, opts=dict(title=title))

    def quiver(self, u, v, title, win=4):
        self.vis.quiver(u, v, win=win, opts=dict(title=title))

