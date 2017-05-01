"""Transforms data from standard P2DDataset/P3DDataset format into something
that SRNN code can understand. Messy because the original H3.6M loader was
messy and I'm too lazy to change the interface."""
import addpaths

import numpy as np
import copy

from p2d_loader import P2DDataset, P3DDataset

rng = np.random.RandomState(1234567890)

global actions


def map_2xp1(d):
    """Maps dictionary of T->[int] into dictionary where each element x in each
    value list has been mapped to both 2*x and 2*x+1. This is meant to mimic
    the way that skeletons are flattened by P2DDataset (joint j will have data
    in the 2*jth and (2*j+1)th columns)."""
    out_d = {}
    for k, l in d.items():
        out_d[k] = new_l = []
        for x in l:
            new_l.extend([2*x, 2*x+1])
    return out_d


def get_feat_ranges(ds_label):
    """Tells loader which features (identified by column indices) to associate
    with which NodeRNNs (identified by name)."""
    nodeFeaturesRanges = {}
    if ds_label == 'ikea':
        # Ikea 7-joint (basically first 8 CPM joints minus the head)
        nodeFeaturesRanges['shoulders'] = [0, 1, 4]
        nodeFeaturesRanges['right_arm'] = [2, 3]
        nodeFeaturesRanges['left_arm'] = [5, 6]
        nodeFeaturesRanges = map_2xp1(nodeFeaturesRanges)
    else:
        raise ValueError(
            "I don't have limb definitions for %s yet (add some here!)" %
            ds_label)
    return nodeFeaturesRanges


def normalizationStats(completeData):
    """Compute mean and standard deviation for full dataset."""
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)
    # Just returns mean and std of data
    return data_mean, data_std


def addNoise(X_old, X_t_1_old, noise=1e-5):
    """Add some (randomly masked) normal noise to tensors."""
    X = copy.deepcopy(X_old)
    X_t_1 = copy.deepcopy(X_t_1_old)

    nodenames = X.keys()
    T1, N1, D1 = X[nodenames[0]].shape
    binomial_prob = rng.binomial(1, 0.5, size=(T1, N1, 1))

    for nm in nodenames:
        noise_to_add = rng.normal(scale=noise, size=X[nm].shape)
        noise_sample = np.repeat(
            binomial_prob, noise_to_add.shape[2], axis=2) * noise_to_add
        X[nm] += noise_sample
        X_t_1[nm][1:, :, :] += noise_sample[:-1, :, :]
    return X, X_t_1


def addNoiseToFeatures(noise=1e-5):
    """Call addNoise on each of the features exposed by this module."""
    global nodeFeatures_noisy, nodeFeatures_t_1_noisy, \
        validate_nodeFeatures_noisy, validate_nodeFeatures_t_1_noisy, \
        forecast_nodeFeatures_noisy, forecast_nodeFeatures_t_1_noisy
    nodeFeatures_noisy, nodeFeatures_t_1_noisy \
        = addNoise(nodeFeatures, nodeFeatures_t_1, noise)
    validate_nodeFeatures_noisy, validate_nodeFeatures_t_1_noisy \
        = addNoise(validate_nodeFeatures, validate_nodeFeatures_t_1, noise)
    forecast_nodeFeatures_noisy, forecast_nodeFeatures_t_1_noisy \
        = addNoise(forecast_nodeFeatures, forecast_nodeFeatures_t_1)


def getlabels(nodeName):
    """Get tensors of output poses which model is supposed to predict under
    various regimes."""
    D = predictFeatures[nodeName].shape[2]
    return predictFeatures[nodeName], \
        validate_predictFeatures[nodeName], \
        forecast_predictFeatures[nodeName], \
        forecast_nodeFeatures[nodeName], \
        D


def getfeatures(nodeName,
                edgeType,
                nodeConnections,
                nodeNames,
                forecast_on_noisy_features=False):
    """Get inputs for a particular node and edge. It's more complex than
    getlabels() because it does a slightly different thing for each
    NodeRNN/EdgeRNN."""
    train_features = getDRAfeatures(nodeName, edgeType, nodeConnections,
                                    nodeNames, nodeFeatures_noisy)
    validate_features = getDRAfeatures(nodeName, edgeType, nodeConnections,
                                       nodeNames, validate_nodeFeatures_noisy)

    forecast_features = []
    if forecast_on_noisy_features:
        forecast_features = getDRAfeatures(
            nodeName, edgeType, nodeConnections, nodeNames,
            forecast_nodeFeatures_noisy)
    else:
        forecast_features = getDRAfeatures(nodeName, edgeType, nodeConnections,
                                           nodeNames, forecast_nodeFeatures)

    return train_features, validate_features, forecast_features


def getDRAfeatures(nodeName, edgeType, nodeConnections, nodeNames,
                   features_to_use):
    """Get input tensor for given combination of node and edge type."""
    if edgeType.split('_')[1] == 'input':
        return features_to_use[nodeName]

    features = []
    nodesConnectedTo = nodeConnections[nodeName]
    for nm in nodesConnectedTo:
        # from nm (connected node) to nodeName
        et1 = nodeNames[nm] + '_' + nodeNames[nodeName]
        # from nodeName to nm (connected node); goes backwards
        et2 = nodeNames[nodeName] + '_' + nodeNames[nm]
        if et1 == et2 and et1 == edgeType:
            # loop from edge of type T to another edge of type T
            f1 = features_to_use[nodeName][:, :, :]
            f2 = features_to_use[nm][:, :, :]
        elif et1 == edgeType:
            # involves this edge type on one side
            f1 = features_to_use[nm][:, :, :]
            f2 = features_to_use[nodeName][:, :, :]
        elif et2 == edgeType:
            # again, involves this edge type on one side
            f1 = features_to_use[nodeName][:, :, :]
            f2 = features_to_use[nm][:, :, :]
        else:
            continue

        # append to feature tensor along axis 2
        if len(features) == 0:
            features = np.concatenate((f1, f2), axis=2)
        else:
            features += np.concatenate((f1, f2), axis=2)

    return features


def cherryPickNodeFeatures(data3DTensor, nodeFeaturesRanges):
    """Divide raw features up into features for each nodeRNN."""
    Features = {}
    nodeNames = nodeFeaturesRanges.keys()
    for nm in nodeNames:
        filterList = []
        for x in nodeFeaturesRanges[nm]:
            filterList.append(x)
        Features[nm] = data3DTensor[:, :, filterList]
    return Features


def addNoiseMalik(X_old, noise=1e-5):
    """Quirky way of adding noise to a tensor. Presumably used by ERD."""
    X = copy.deepcopy(X_old)
    T1, N1, D1 = X.shape
    binomial_prob = rng.binomial(1, 0.5, size=(T1, N1, 1))
    noise_to_add = rng.normal(scale=noise, size=X.shape)
    noise_sample = np.repeat(
        binomial_prob, noise_to_add.shape[2], axis=2) * noise_to_add
    X += noise_sample
    return X


##############################################
# Convenience functions for getting ERD data #
##############################################


def getMalikFeatures(noise=1e-5):
    return addNoiseMalik(malikTrainFeatures, noise=noise), malikPredictFeatures


def getMalikValidationFeatures(noise=1e-5):
    return addNoiseMalik(
        validate_malikTrainFeatures,
        noise=noise), validate_malikPredictFeatures


def getMalikTrajectoryForecasting(noise=1e-5):
    return trX_forecast_malik, trY_forecast_malik


#############################################################################
# Globals which are used to configure this module (once runall() is called) #
#############################################################################

T = 150
motion_prefix = 50
motion_suffix = 100
train_for = 'final'
crf_file = ''
ds_label = 'this is invalid; you should define it properly from runPP.py'
ds_path = None
ds_is_3d = None
dataset = None
# drop_features and dimensions_to_ignore only need to be here so that DRA.py
# (in NeuralModels) can find them
drop_features = False
dimensions_to_ignore = np.asarray([])

##################################################
# Setup code for datasets exposed by this module #
##################################################


def runall():
    global data_stats, data3Dtensor, Y3Dtensor, validate3Dtensor, \
        validateY3Dtensor, trX_forecast, trY_forecast, malikTrainFeatures, \
        malikPredictFeatures, validate_malikTrainFeatures, \
        validate_malikPredictFeatures, trX_forecast_malik, \
        trY_forecast_malik, data_mean, data_std, \
        new_idx, nodeFeatures, predictFeatures, validate_nodeFeatures, \
        validate_predictFeatures, forecast_nodeFeatures, \
        forecast_predictFeatures, trainSubjects, new_idx, \
        validateSubject, actions, nodeFeatures_t_1, nodeFeaturesRanges, \
        validate_nodeFeatures_t_1, forecast_nodeFeatures_t_1, dataset

    if ds_is_3d:
        dataset = P3DDataset(ds_path)
    else:
        # I don't know why, but this still requires a seq_length for some
        # reason
        # XXX: need to get rid of IkeaDB-specialised head removal crap. If head
        # sucks then it should be stripped from .h5 in the first place.
        dataset = P2DDataset(ds_path, seq_length=32, remove_head=True)

    # Compute training data mean
    all_data_tensor, _ = dataset.get_ds_for_train(train=True,
                                                  seq_length=64,
                                                  discard_shorter=64,
                                                  gap=64)
    # we flatten this into (N*T)*D tensor, rather than N*T*D
    all_data_tensor = all_data_tensor.reshape((-1, all_data_tensor.shape[2]))
    data_mean, data_std = normalizationStats(all_data_tensor)
    data_stats = {}
    data_stats['mean'] = data_mean
    data_stats['std'] = data_std
    print T

    def divvy_up_like_sTS(tensor):
        """Splits an an N*(T+2)*D tensor into three T*N*D tensors mimicking the
        ones produced by sampleTrainSequences."""
        tensor = tensor.transpose((1, 0, 2))
        assert tensor.shape[0] == T + 2, tensor.shape
        return tensor[1:-1], tensor[2:], tensor[0:-2]

    # 2 normalized 3D tensor for training and validation
    d4t_kwargs = {
        'seq_length': T + 2,
        'gap': 12,
        # make sure it doesn't return any truncated sequences
        'discard_shorter': T + 2
    }
    data_train_Tp2, mt = dataset.get_ds_for_train(train=True, **d4t_kwargs)
    data_val_Tp2, mv = dataset.get_ds_for_train(train=False, **d4t_kwargs)
    if not np.all(mt != 0) or not np.all(mv != 0):
        print('WARNING: there are (ignored) masked steps in one of the '
              'training sets!')
    data3Dtensor, Y3Dtensor, data3Dtensor_t_1 \
        = divvy_up_like_sTS(data_train_Tp2)
    validate3Dtensor, validateY3Dtensor, validate3Dtensor_t_1 \
        = divvy_up_like_sTS(data_val_Tp2)

    print 'Training data stats (T,N,D) is ', data3Dtensor.shape
    print 'Training data stats (T,N,D) is ', validate3Dtensor.shape

    # Generate normalized data for trajectory forecasting
    fcst_len = motion_prefix + motion_suffix + 1

    def divvy_up_like_gFE(tensor):
        """Divides a big N*(prefix+suffix+1)*D tensor into tensors of size
        prefix*N*D, prefix*N*D, and suffix*N*D in the same manner as
        generateForecastingExamples."""
        tensor = tensor.transpose((1, 0, 2))
        assert tensor.shape[0] == fcst_len, tensor.shape
        return tensor[1:1+motion_prefix], tensor[:motion_prefix], \
            tensor[motion_prefix+1:]
    data_val_fcst, mvg \
        = dataset.get_ds_for_train(train=False, seq_length=fcst_len, gap=3,
                                   discard_shorter=fcst_len)
    if not np.all(mvg != 0):
        print('WARNING: there are ignored (masked) steps in the forecasting'
              'data')
    trX_forecast, trX_forecast_t_1, trY_forecast \
        = divvy_up_like_gFE(data_val_fcst)

    nodeFeaturesRanges = get_feat_ranges(ds_label)
    # validation check to make sure everything is accounted for
    data_cols = set(range(trX_forecast.shape[-1]))
    used_cols = {c for clist in nodeFeaturesRanges.values() for c in clist}
    unused_cols = data_cols - used_cols
    if unused_cols:
        col_str = ','.join(map(str, unused_cols))
        n_unused = len(unused_cols)
        print('WARNING: %d columns are unused: %s' % (n_unused, col_str))
    assert not unused_cols, "Can't continue with unused feature columns"
    new_idx = np.asarray(sorted(used_cols))

    # Create training and validation features for DRA
    nodeFeatures = cherryPickNodeFeatures(data3Dtensor, nodeFeaturesRanges)
    nodeFeatures_t_1 = cherryPickNodeFeatures(data3Dtensor_t_1,
                                              nodeFeaturesRanges)
    validate_nodeFeatures = cherryPickNodeFeatures(validate3Dtensor,
                                                   nodeFeaturesRanges)
    validate_nodeFeatures_t_1 = cherryPickNodeFeatures(validate3Dtensor_t_1,
                                                       nodeFeaturesRanges)
    forecast_nodeFeatures = cherryPickNodeFeatures(trX_forecast,
                                                   nodeFeaturesRanges)
    forecast_nodeFeatures_t_1 = cherryPickNodeFeatures(trX_forecast_t_1,
                                                       nodeFeaturesRanges)

    predictFeatures = cherryPickNodeFeatures(Y3Dtensor, nodeFeaturesRanges)
    validate_predictFeatures = cherryPickNodeFeatures(validateY3Dtensor,
                                                      nodeFeaturesRanges)
    forecast_predictFeatures = cherryPickNodeFeatures(trY_forecast,
                                                      nodeFeaturesRanges)

    # Create training and validation features for Malik's LSTM model
    malikTrainFeatures = data3Dtensor
    malikPredictFeatures = Y3Dtensor
    validate_malikTrainFeatures = validate3Dtensor
    validate_malikPredictFeatures = validateY3Dtensor
    trX_forecast_malik = trX_forecast
    trY_forecast_malik = trY_forecast
