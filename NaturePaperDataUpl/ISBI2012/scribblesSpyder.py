#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:19:42 2017

@author: ahmadnish
"""

import sys
sys.path.append('/Users/ahmadnish/dev/bld/nifty/python')

import numpy
import scipy
import pylab

import nifty
import nifty.graph.rag
import nifty.segmentation


import skimage.filters       # filters
import skimage.segmentation  # Superpixels
import skimage.data          # Data
import skimage.color         # rgb2Gray
import matplotlib


# shape = [3,3]
# g = nifty.graph.undirectedGridGraph(shape)

# print(g.numberOfNodes)
# print(g.numberOfEdges)

# rag = nifty.graph.rag.gridRag(g)


img = skimage.data.coins()
print(img.shape)
# print(img.dtype)
# print(img.max(),img.min())

overseg = skimage.segmentation.slic(img, n_segments=2000,
    compactness=0.1, sigma=1)

#overseg = nifty.segmentation.distanceTransformWatersheds(img, threshold=0.3)


print(overseg.shape)
rag = nifty.graph.rag.gridRag(overseg)
print(rag)

a,b = pylab.rcParams['figure.figsize']
pylab.rcParams['figure.figsize'] = 1.5*a, 1.5*b

f = pylab.figure()
f.add_subplot(2, 2, 1)
pylab.imshow(img, cmap='gray')
pylab.title('Raw Data')

f.add_subplot(2, 2, 2)
# bImg = nifty.segmentation.markBoundaries(img, overseg, color=(1,0,0))
pylab.imshow(overseg, cmap='gray')
pylab.title('Superpixels')



pylab.show()
