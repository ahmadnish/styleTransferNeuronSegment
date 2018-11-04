"""
ISBI 2012 Simple 2D Multicut Pipeline
======================================

Here we segment neuro data as in  :cite:`beier_17_multicut`.
In fact, this is a simplified version of :cite:`beier_17_multicut`.
We start from an distance transform watershed
over-segmentation.
We compute a RAG and features for all edges.
Next, we learn the edge probabilities
with a random forest classifier.
The predicted edge probabilities are
fed into multicut objective.
This is optimized with an ILP solver (if available).
This results into a ok-ish learned segmentation
for the ISBI 2012 dataset.

This example will download a about 400 MB large zip file
with the dataset and precomputed results from :cite:`beier_17_multicut`



"""


# multi purpose
import numpy
import numpy as np
import scipy

# plotting
import pylab

# to download data and unzip it
import os
import urllib.request
import zipfile

# to read the tiff files
import skimage.io
import skimage.filters
import skimage.morphology

# classifier
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('/Users/ahmadnish/dev3/bld/nifty/python')

# needed parts of nifty
import nifty
import nifty.segmentation
import nifty.filters
import nifty.graph.rag
import nifty.ground_truth
import nifty.graph.opt.multicut
#############################################################
# Download  ISBI 2012:
# =====================
# Download the  ISBI 2012 dataset 
# and precomputed results form :cite:`beier_17_multicut`
# and extract it in-place.
# fname = "data.zip"
# url = "http://files.ilastik.org/multicut/NaturePaperDataUpl.zip"
# if not os.path.isfile(fname):
#     urllib.request.urlretrieve(url, fname)
#     zip = zipfile.ZipFile(fname)
#     zip.extractall()

#############################################################
# Setup Datasets:
# =================
# load ISBI 2012 raw and probabilities
# for train and test set
# and the ground-truth for the train set
# debugging: only use part of the training images

# we use part of the training images as ground truth
split = 24
lim = 24
stylize = 'on'
shuffle = 'off'
rawDsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/raw_train.tif')[0:lim],
    'test' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/raw_train.tif')[split:],
}

gtDsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/groundtruth.tif')[0:lim],
    'test'  : skimage.io.imread('NaturePaperDataUpl/ISBI2012/groundtruth.tif')[split:]
}

if stylize == 'on':

    train = []
    styles = [3, 10, 19, 21]
    # styles = range(24)

    for i in range(lim):

        im = []

        im.append(rawDsets['train'][i])
        for j in styles:
            im.append(skimage.io.imread("output/nish_output_paintings/stylized-train{}-style{}-75.jpg".format(i,j)))

        im = np.stack(im, axis = 0)

        train.append(im)



    trainset = np.concatenate(train, axis = 0)

    gt_train = []
    for i in range(lim):
        temp = [gtDsets['train'][i] for j in range(len(styles)+1)]
        temp = np.stack(temp, axis = 0)
        gt_train.append(temp)

    gtTrain = np.concatenate(gt_train, axis = 0)

    print(gtTrain.shape)

    # if shuffle == 'on':
    #     perm = np.arange(trainset.shape[0])
    #     np.random.shuffle(perm)
    # else:
    #     perm = np.arange(trainset.shape[0])

    rawDsets['train'] = trainset
    gtDsets['train'] = gtTrain

computedData = {
    'train' : [{} for z in range(rawDsets['train'].shape[0])],
    'test'  : [{} for z in range(rawDsets['test'].shape[0])]
}

plot_multiplier = 15
plotting = 'off'

assert gtDsets['train'].shape == rawDsets['train'].shape
assert gtDsets['test'].shape == rawDsets['test'].shape
print("train dset size:{}".format(rawDsets['train'].shape))
print("gtDsets size:{}".format(gtDsets['train'].shape))
print("test dset size:{}".format(rawDsets['test'].shape))
print("gtDsets size:{}".format(gtDsets['test'].shape))

#############################################################
# Helper Functions:
# ===================
# Function to compute features for a RAG
# (used later)
def computeFeatures(raw, rag):

    uv = rag.uvIds()
    nrag = nifty.graph.rag

    # list of all edge features we fill 
    feats = []

    # helper function to convert 
    # node features to edge features
    def nodeToEdgeFeat(nodeFeatures):
        uF = nodeFeatures[uv[:,0], :]
        vF = nodeFeatures[uv[:,1], :]
        feats = [ numpy.abs(uF-vF), uF + vF, uF *  vF,
                 numpy.minimum(uF,vF), numpy.maximum(uF,vF)]
        return numpy.concatenate(feats, axis=1)


    # accumulate features from raw data
    fRawEdge, fRawNode = nrag.accumulateStandartFeatures(rag=rag, data=raw,
        minVal=0.0, maxVal=255.0, numberOfThreads=1)
    feats.append(fRawEdge)
    feats.append(nodeToEdgeFeat(fRawNode))

    # accumulate node and edge features from
    # superpixels geometry 
    fGeoEdge = nrag.accumulateGeometricEdgeFeatures(rag=rag, numberOfThreads=1)
    feats.append(fGeoEdge)

    fGeoNode = nrag.accumulateGeometricNodeFeatures(rag=rag, numberOfThreads=1)
    feats.append(nodeToEdgeFeat(fGeoNode))

    return numpy.concatenate(feats, axis=1)


#############################################################
#  Over-segmentation, RAG & Extract Features:
#  ============================================
# 
#  Compute:
#   *    Over-segmentation  with distance transform watersheds.
#   *    Construct a region adjacency graph (RAG)
#   *    Extract features for all edges in the graph
#   *    Map the ground truth to the edges in the graph.
#        (only for the training set)
#
print('computing ...')
for ds in ['train', 'test']:
    
    rawDset = rawDsets[ds]
    gtDset = gtDsets[ds]
    dataDset = computedData[ds]

    # for each slice
    for z in range(rawDset.shape[0]):   
        
        
        data = dataDset[z]

        # get raw slice
        raw  = rawDset[z, ... ]

        
        fraw = skimage.filters.frangi(raw)
        data['fraw'] = fraw
        # fraw = raw
        # oversementation
        overseg = nifty.segmentation.distanceTransformWatersheds(fraw, threshold=0.3)
        overseg -= 1
        data['overseg'] = overseg

        # region adjacency graph
        rag = nifty.graph.rag.gridRag(overseg)
        data['rag'] = rag

        # compute features
        features = computeFeatures(raw=raw, rag=rag)

        data['features'] = features

        # map the gt to edge

        # the gt is on membrane level
        # 0 at membranes pixels
        # 1 at non-membrane pixels
        gtImage = gtDset[z, ...] 

        # local maxima seeds
        seeds = nifty.segmentation.localMaximaSeeds(gtImage)

        data['partialGt'] = seeds
        # growing map
        growMap = nifty.filters.gaussianSmoothing(1.0-gtImage, 1.0)
        growMap += 0.1*nifty.filters.gaussianSmoothing(1.0-gtImage, 6.0)
        gt = nifty.segmentation.seededWatersheds(growMap, seeds=seeds)

        if(ds == 'test'):
            data['seeds'] = seeds
            data['gt'] = gt

        # map the gt to the edges
        overlap = nifty.ground_truth.overlap(segmentation=overseg, 
                                   groundTruth=gt)

        # edge gt
        edgeGt = overlap.differentOverlaps(rag.uvIds())
        data['edgeGt'] = edgeGt


            # plot each 14th 
        if z  % plot_multiplier == 0 and plotting == 'on' :
            figure = pylab.figure()
            figure.suptitle('Training Set Slice %d'%z, fontsize=20)

            #fig = matplotlib.pyplot.gcf()
            figure.set_size_inches(18.5, 10.5)

            figure.add_subplot(3, 2, 1)
            pylab.imshow(raw, cmap='gray')
            pylab.title("Raw data %s"%(ds))


            figure.add_subplot(3, 2, 3)
            pylab.imshow(nifty.segmentation.segmentOverlay(raw, overseg, 0.2, thin=False))
            pylab.title("Superpixels %s"%(ds))

            figure.add_subplot(3, 2, 4)
            pylab.imshow(seeds, cmap=nifty.segmentation.randomColormap(zeroToZero=True))
            pylab.title("Partial ground truth %s" %(ds))

            figure.add_subplot(3, 2, 5)
            pylab.imshow(nifty.segmentation.segmentOverlay(raw, gt, 0.2, thin=False))
            pylab.title("Dense ground truth %s" %(ds))
            pylab.tight_layout()
            pylab.show()


print('done')
#############################################################
# Build the training set:
# ===========================
# We only use high confidence boundaries.
dataDset = computedData[ds]
trainingSet = {'features':[],'labels':[]}

for ds in ['train']:
    
    rawDset = rawDsets[ds]
    gtDset = gtDsets[ds]
    dataDset = computedData[ds]

    # for each slice
    for z in range(rawDset.shape[0]):   

        data = dataDset[z]

        rag = data['rag']
        edgeGt = data['edgeGt']    
        features = data['features']

        # we use only edges which have
        # a high certainty
        where1 = numpy.where(edgeGt > 0.8)[0]
        where0 = numpy.where(edgeGt < 0.2)[0]

        trainingSet['features'].append(features[where0,:])
        trainingSet['features'].append(features[where1,:])
        trainingSet['labels'].append(numpy.zeros(len(where0)))
        trainingSet['labels'].append(numpy.ones(len(where1)))

features = numpy.concatenate(trainingSet['features'], axis=0)
labels = numpy.concatenate(trainingSet['labels'], axis=0)

#############################################################
# Train the random forest (RF):
# ===============================
print(features.shape, labels.shape)
rf = RandomForestClassifier(n_estimators=200, oob_score=True)
rf.fit(features, labels)
print("OOB SCORE",rf.oob_score_)



#############################################################'
# Predict Edge Probabilities & Optimize Multicut Objective:
# ===========================================================
#
# Predict the edge probabilities with the learned
# random forest classifier.
# Set up a multicut objective and find the argmin
# with an ILP solver (if available).

def saveimage(z):
    """Transforms the pytorch tensor into an image

    Arguments:
        tensor {torch.float} -- the original loaded input_image, now the stylized image

    Keyword Arguments:
        save_name {str} -- name of the file  (default: {"test"})
    """
    if os.path.exists('output/testing_testRF{}.jpg'.format(z)):
        i = 0
        while os.path.exists('output/testing_testRF{}-{:d}.jpg'.format(z, i)):
            i += 1
        save_name = 'output/testing_testRF{}-{:d}.jpg'.format(z, i)
    else:
        save_name = "output/testing_testRF{}.jpg".format(z)


    pylab.savefig(save_name)

for ds in ['test']:
    
    rawDset = rawDsets[ds]
    gtDset = gtDsets[ds]
    dataDset = computedData[ds]

    # for each slice
    argResult = []
    for z in range(rawDset.shape[0]):   

        
        data = dataDset[z]

        raw = rawDset[z,...]
        fraw = data['fraw']
        overseg = data['overseg']
        rag = data['rag']
        edgeGt = data['edgeGt']    
        features = data['features']
        gt = data['gt']
        predictions = rf.predict_proba(features)[:,1]


        # setup multicut objective
        MulticutObjective = rag.MulticutObjective

        eps =  0.00001
        p1 = numpy.clip(predictions, eps, 1.0 - eps) 
        weights = numpy.log((1.0-p1)/p1)


        

        objective = MulticutObjective(rag, weights)

        # do multicut obtimization 
        if nifty.Configuration.WITH_CPLEX:
            solver = MulticutObjective.multicutIlpCplexFactory().create(objective)
        elif nifty.Configuration.WITH_GLPK:
            solver = MulticutObjective.multicutIlpGurobiFactory().create(objective)
        elif nifty.Configuration.WITH_GLPK:
            solver = MulticutObjective.multicutIlpGlpkFactory().create(objective)
        else:
            solver = MulticutObjective.greedyAdditiveFactory().create(objective)


        arg = solver.optimize(visitor=MulticutObjective.verboseVisitor())
        result = nifty.graph.rag.projectScalarNodeDataToPixels(rag, arg)

        argResult.append(arg)

        # plot for each 14th slice

        figure = pylab.figure()
        
        figure.suptitle('Test Set Results Slice %d'%z, fontsize=20)

        #fig = matplotlib.pyplot.gcf()
        figure.set_size_inches(18.5, 10.5)

        figure.add_subplot(3, 2, 1)
        pylab.imshow(raw, cmap='gray')
        pylab.title("Raw data %s"%(ds))

        figure.add_subplot(3, 2, 2)
        pylab.imshow(fraw, cmap='gray')
        pylab.title("fRaw data %s"%(ds))


        figure.add_subplot(3, 2, 3)
        pylab.imshow(nifty.segmentation.segmentOverlay(raw, overseg, 0.2, thin=False))
        pylab.title("Superpixels %s"%(ds))

        figure.add_subplot(3, 2, 4)
        pylab.imshow(result, cmap=nifty.segmentation.randomColormap())
        pylab.title("Result seg. %s" %(ds))

        figure.add_subplot(3, 2, 5)
        pylab.imshow(nifty.segmentation.segmentOverlay(raw, result, 0.2, thin=False))
        pylab.title("Result seg. %s" %(ds))
        pylab.tight_layout()
        

        figure.add_subplot(3, 2, 6)
        pylab.imshow(nifty.segmentation.segmentOverlay(raw, gt, 0.2, thin=False))
        pylab.title("Ground Truth")
        pylab.tight_layout()
        saveimage(z)

from nifty import ground_truth

ds = 'test'
rawDset = rawDsets[ds]
dataDset = computedData[ds]
# dimensions: test_images, random_forests, benchmarks
error = numpy.zeros([rawDset.shape[0], 1, 2])
# for each slice
for z in range(rawDset.shape[0]):
        
    data = dataDset[z]
    partialGt = data['partialGt']
    rag = data['rag']
    for r in range(1):
               
        seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, argResult[z])    
        randError = ground_truth.RandError(partialGt, seg, ignoreDefaultLabel = True)
        variationError = ground_truth.VariationOfInformation(partialGt, seg, ignoreDefaultLabel = True)
        error[z,r,:] = [randError.error, variationError.value]

mean_error = numpy.mean(error, axis = 0)
print(mean_error)