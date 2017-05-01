#!/usr/bin/env python2
"""Does something similar to hyperParameterTuning.py and trainDRA.py, but
specialised to my pose prediction data format.

Note that this does *NOT* rely on hyperParameterTuning.py or trainDRA.py. You
should not need to make changes to this file for this one to work!"""
import argparse
import os
import sys

from neuralmodels.costs import euclidean_loss
from neuralmodels.layers import LSTM, multilayerLSTM, FCLayer, \
    TemporalInputFeatures
from neuralmodels.loadcheckpoint import load, loadDRA
from neuralmodels.models import DRA, noisyRNN
from neuralmodels.updates import Momentum
import numpy as np
import theano
from theano import tensor as T

from unNormalizeData import unNormalizeData
import crfproblems.stdpp.processdata as poseDataset
import readCRFgraph as graph

rng = np.random.RandomState(1234567890)


def get_default_args():
    train_model = 'srnn'  # also try 'lstm3lr' or 'erd'

    base_dir = open('basedir', 'r').readline().strip()

    # Hyper parameters for training S-RNN
    params = {}

    if train_model == 'srnn':
        # These hyperparameters are OKAY to tweak. They will affect training,
        # convergence etc.
        params['initial_lr'] = 1e-3
        # Decrease learning rate after these many iterations
        params['decay_schedule'] = [1.5e3, 4.5e3]
        # Multiply the current learning rate by this factor
        params['decay_rate_schedule'] = [0.1, 0.1]
        params['lstm_init'] = 'uniform'  # Initialization of lstm weights
        params['fc_init'] = 'uniform'  # Initialization of FC layer weights
        params['clipnorm'] = 25.0
        params['use_noise'] = 1
        # Add noise after these many iterations
        params['noise_schedule'] = [250, 0.5e3, 1e3, 1.3e3, 2e3, 2.5e3, 3.3e3]
        # Variance of noise to add
        params['noise_rate_schedule'] = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
        params['momentum'] = 0.99
        params['g_clip'] = 25.0
        params['truncate_gradient'] = 10  # 100
        params['sequence_length'] = 150  # Length of each sequence fed to RNN
        params['sequence_overlap'] = 50
        params['batch_size'] = 100
        params['lstm_size'] = 10  # 512
        params['node_lstm_size'] = 10  # 512
        params['fc_size'] = 10  # 256
        # Save the model after every 250 iterations
        params['snapshot_rate'] = 250
        params['train_for'] = 'final'
        # Possible options are ['eating','smoking','discussion','final','']
        # '': Use this for validation and hyperparameter tuning
        # 'final': Will train on activities {eating, smoking, walking,
        # discussion}
        # 'eating': Will only train on eating activity
        # Look Process data file for more details

        # Tweak these hyperparameters only if you want to try out new models
        # etc. This is only for 'Advanced' users
        params['use_pretrained'] = 0
        params['iter_to_load'] = 2500
        params['model_to_train'] = 'dra'
        params['crf'] = ''
        params['weight_decay'] = 0.0
        params['dra_type'] = 'simple'

    elif train_model == 'lstm3lr' or train_model == 'erd':
        # These hyperparameters are OKAY to tweak. They will affect training,
        # convergence etc.
        params['truncate_gradient'] = 100
        params['sequence_length'] = 150
        params['sequence_overlap'] = 50
        params['batch_size'] = 100
        # This parameter is same as the one used by Fragkiadaki et al. ICCV'15
        params['lstm_size'] = 1000
        # This parameter is same as the one used by Fragkiadaki et al. ICCV'15
        params['fc_size'] = 500
        params['use_noise'] = 1
        # Add noise after these many iterations
        params['noise_schedule'] = [250, 0.5e3, 1e3, 1.3e3, 2e3, 2.5e3, 3.3e3]
        # Variance of noise to add
        params['noise_rate_schedule'] = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
        params['initial_lr'] = 1e-3
        # Decrease learning rate after these many iterations
        params['decay_schedule'] = [1.5e3, 4.5e3]
        # Multiply the current learning rate by this factor
        params['decay_rate_schedule'] = [0.1, 0.1]
        params['train_for'] = 'final'
        params['clipnorm'] = 25.0

        params['use_pretrained'] = 0
        params['iter_to_load'] = 1250
        if train_model == 'lstm3lr':
            params['model_to_train'] = 'lstm'
        else:
            params['model_to_train'] = 'malik'
        params['snapshot_rate'] = 250
        params['crf'] = ''
        params['weight_decay'] = 0.0

    # Setting directory to dump trained models and then executing trainDRA.py
    params['checkpoint_path'] \
        = 'checkpoints_{0}_T_{2}_bs_{1}_tg_{3}_ls_{4}_fc_{5}_demo'.format(
            params['model_to_train'], params['batch_size'],
            params['sequence_length'], params['truncate_gradient'],
            params['lstm_size'], params['fc_size'])
    path_to_checkpoint = base_dir + '/{0}/'.format(params['checkpoint_path'])
    if not os.path.exists(path_to_checkpoint):
        os.mkdir(path_to_checkpoint)
    print 'Dir: {0}'.format(path_to_checkpoint)
    default_args = []
    for k in params.keys():
        default_args.append('--{0}'.format(k))
        if not isinstance(params[k], list):
            default_args.append(str(params[k]))
        else:
            for x in params[k]:
                default_args.append(str(x))
    return default_args


parser = argparse.ArgumentParser()
parser.add_argument('--decay_type', type=str, default='schedule')
parser.add_argument('--decay_after', type=int, default=-1)
parser.add_argument('--initial_lr', type=float, default=1e-3)
parser.add_argument('--learning_rate_decay', type=float, default=1.0)
parser.add_argument('--decay_schedule', nargs='*', type=float)
parser.add_argument('--decay_rate_schedule', nargs='*', type=float)
parser.add_argument('--node_lstm_size', type=int, default=10)
parser.add_argument('--lstm_size', type=int, default=10)
parser.add_argument('--fc_size', type=int, default=500)
parser.add_argument('--lstm_init', type=str, default='uniform')
parser.add_argument('--fc_init', type=str, default='uniform')
parser.add_argument('--snapshot_rate', type=int, default=1)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=3000)
parser.add_argument('--clipnorm', type=float, default=25.0)
parser.add_argument('--use_noise', type=int, default=1)
parser.add_argument('--noise_schedule', nargs='*', type=float)
parser.add_argument('--noise_rate_schedule', nargs='*', type=float)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--g_clip', type=float, default=25.0)
parser.add_argument('--truncate_gradient', type=int, default=50)
parser.add_argument('--use_pretrained', type=int, default=0)
parser.add_argument('--iter_to_load', type=int, default=None)
parser.add_argument('--model_to_train', type=str, default='dra')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint')
parser.add_argument('--sequence_length', type=int, default=150)
parser.add_argument('--sequence_overlap', type=int, default=50)
parser.add_argument('--maxiter', type=int, default=15000)
parser.add_argument('--crf', type=str, default='')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--train_for', type=str, default='validate')
parser.add_argument('--dra_type', type=str, default='simple')
parser.add_argument(
    '--3d',
    action='store_true',
    dest='is_3d',
    default=False,
    help='read 3D features from this dataset (otherwise read 2D ones)')
parser.add_argument('dataset_path', help='path to .h5 containing poses')
parser.add_argument('output_dir', help='where to store snapshots and poses')
args = parser.parse_args(get_default_args() + sys.argv[1:])

print args
if args.use_pretrained:
    print 'Loading pre-trained model with iter={0}'.format(args.iter_to_load)
gradient_method = Momentum(momentum=args.momentum)

# Loads H3.6m dataset
poseDataset.T = args.sequence_length
poseDataset.delta_shift = args.sequence_length - args.sequence_overlap
poseDataset.num_forecast_examples = 24
poseDataset.train_for = args.train_for
poseDataset.crf_file = './crfproblems/stdpp/crf' + args.crf
poseDataset.ds_path = args.dataset_path
poseDataset.ds_is_3d = args.is_3d
poseDataset.runall()


def saveForecastedMotion(forecast, path, prefix='ground_truth_forecast_N_'):
    T = forecast.shape[0]
    N = forecast.shape[1]
    D = forecast.shape[2]
    for j in range(N):
        motion = forecast[:, j, :]
        f = open('{0}{2}{1}'.format(path, j, prefix), 'w')
        for i in range(T):
            st = ''
            for k in range(D):
                st += str(motion[i, k]) + ','
            st = st[:-1]
            f.write(st + '\n')
        f.close()


def DRAmodelRegression(nodeList, edgeList, edgeListComplete, edgeFeatures,
                       nodeFeatureLength, nodeToEdgeConnections):

    edgeRNNs = {}
    edgeNames = edgeList

    for em in edgeNames:
        inputJointFeatures = edgeFeatures[em]
        LSTMs = [
            LSTM(
                'tanh',
                'sigmoid',
                args.lstm_init,
                truncate_gradient=args.truncate_gradient,
                size=args.lstm_size,
                rng=rng,
                g_low=-args.g_clip,
                g_high=args.g_clip)
        ]
        edgeRNNs[em] = [
            TemporalInputFeatures(inputJointFeatures),
            # AddNoiseToInput(rng=rng),
            FCLayer('rectify', args.fc_init, size=args.fc_size, rng=rng),
            FCLayer('linear', args.fc_init, size=args.fc_size, rng=rng),
            multilayerLSTM(
                LSTMs,
                skip_input=True,
                skip_output=True,
                input_output_fused=True)
        ]

    nodeRNNs = {}
    nodeTypes = nodeList.keys()
    nodeLabels = {}
    for nm in nodeTypes:
        num_classes = nodeList[nm]
        LSTMs = [
            LSTM(
                'tanh',
                'sigmoid',
                args.lstm_init,
                truncate_gradient=args.truncate_gradient,
                size=args.node_lstm_size,
                rng=rng,
                g_low=-args.g_clip,
                g_high=args.g_clip)
        ]
        nodeRNNs[nm] = [
            multilayerLSTM(
                LSTMs,
                skip_input=True,
                skip_output=True,
                input_output_fused=True),
            FCLayer('rectify', args.fc_init, size=args.fc_size, rng=rng),
            FCLayer('rectify', args.fc_init, size=100, rng=rng),
            FCLayer('linear', args.fc_init, size=num_classes, rng=rng)
        ]
        em = nm + '_input'
        edgeRNNs[em] = [
            TemporalInputFeatures(nodeFeatureLength[nm]),
            # AddNoiseToInput(rng=rng),
            FCLayer('rectify', args.fc_init, size=args.fc_size, rng=rng),
            FCLayer('linear', args.fc_init, size=args.fc_size, rng=rng)
        ]
        nodeLabels[nm] = T.tensor3(dtype=theano.config.floatX)
    learning_rate = T.scalar(dtype=theano.config.floatX)
    dra = DRA(edgeRNNs,
              nodeRNNs,
              nodeToEdgeConnections,
              edgeListComplete,
              euclidean_loss,
              nodeLabels,
              learning_rate,
              clipnorm=args.clipnorm,
              update_type=gradient_method,
              weight_decay=args.weight_decay)
    return dra


def DRAmodelRegressionNoEdge(nodeList, edgeList, edgeListComplete,
                             edgeFeatures, nodeFeatureLength,
                             nodeToEdgeConnections):

    edgeRNNs = {}
    edgeNames = edgeList

    for em in edgeNames:
        inputJointFeatures = edgeFeatures[em]
        LSTMs = [
            LSTM(
                'tanh',
                'sigmoid',
                args.lstm_init,
                truncate_gradient=args.truncate_gradient,
                size=args.lstm_size,
                rng=rng,
                g_low=-args.g_clip,
                g_high=args.g_clip)
            # LSTM('tanh','sigmoid',args.lstm_init,truncate_gradient=args.truncate_gradient,size=args.lstm_size,rng=rng,g_low=-args.g_clip,g_high=args.g_clip)
        ]
        edgeRNNs[em] = [
            TemporalInputFeatures(inputJointFeatures),
            # AddNoiseToInput(rng=rng),
            FCLayer('rectify', args.fc_init, size=args.fc_size, rng=rng),
            FCLayer('linear', args.fc_init, size=args.fc_size, rng=rng)
            # multilayerLSTM(LSTMs,skip_input=True,skip_output=True,input_output_fused=True)
        ]

    nodeRNNs = {}
    nodeTypes = nodeList.keys()
    nodeLabels = {}
    for nm in nodeTypes:
        num_classes = nodeList[nm]
        LSTMs = [
            LSTM(
                'tanh',
                'sigmoid',
                args.lstm_init,
                truncate_gradient=args.truncate_gradient,
                size=args.node_lstm_size,
                rng=rng,
                g_low=-args.g_clip,
                g_high=args.g_clip)
        ]
        nodeRNNs[nm] = [
            multilayerLSTM(
                LSTMs,
                skip_input=True,
                skip_output=True,
                input_output_fused=True),
            FCLayer('rectify', args.fc_init, size=args.fc_size, rng=rng),
            FCLayer('rectify', args.fc_init, size=100, rng=rng),
            FCLayer('linear', args.fc_init, size=num_classes, rng=rng)
        ]
        em = nm + '_input'
        edgeRNNs[em] = [
            TemporalInputFeatures(nodeFeatureLength[nm]),
            # AddNoiseToInput(rng=rng),
            FCLayer('rectify', args.fc_init, size=args.fc_size, rng=rng),
            FCLayer('linear', args.fc_init, size=args.fc_size, rng=rng)
        ]
        nodeLabels[nm] = T.tensor3(dtype=theano.config.floatX)
    learning_rate = T.scalar(dtype=theano.config.floatX)
    dra = DRA(edgeRNNs,
              nodeRNNs,
              nodeToEdgeConnections,
              edgeListComplete,
              euclidean_loss,
              nodeLabels,
              learning_rate,
              clipnorm=args.clipnorm,
              update_type=gradient_method,
              weight_decay=args.weight_decay)
    return dra


def DRAmodelRegression_RNNatEachNode(nodeList, edgeList, edgeListComplete,
                                     edgeFeatures, nodeFeatureLength,
                                     nodeToEdgeConnections):

    edgeRNNs = {}
    edgeNames = edgeList

    for em in edgeNames:
        inputJointFeatures = edgeFeatures[em]
        LSTMs = [
            LSTM(
                'tanh',
                'sigmoid',
                args.lstm_init,
                truncate_gradient=args.truncate_gradient,
                size=args.lstm_size,
                rng=rng,
                g_low=-args.g_clip,
                g_high=args.g_clip)
        ]
        edgeRNNs[em] = [
            TemporalInputFeatures(inputJointFeatures),
            FCLayer('rectify', args.fc_init, size=args.fc_size, rng=rng),
            FCLayer('linear', args.fc_init, size=args.fc_size, rng=rng),
            multilayerLSTM(
                LSTMs,
                skip_input=True,
                skip_output=True,
                input_output_fused=True)
        ]

    nodeRNNs = {}
    nodeTypes = nodeList.keys()
    nodeLabels = {}
    for nm in nodeTypes:
        num_classes = nodeList[nm]
        LSTMs = [
            LSTM(
                'tanh',
                'sigmoid',
                args.lstm_init,
                truncate_gradient=args.truncate_gradient,
                size=args.node_lstm_size,
                rng=rng,
                g_low=-args.g_clip,
                g_high=args.g_clip)
        ]
        nodeRNNs[nm] = [
            multilayerLSTM(
                LSTMs,
                skip_input=True,
                skip_output=True,
                input_output_fused=True),
            FCLayer('rectify', args.fc_init, size=args.fc_size, rng=rng),
            FCLayer('rectify', args.fc_init, size=100, rng=rng),
            FCLayer('linear', args.fc_init, size=num_classes, rng=rng)
        ]
        em = nm + '_input'
        LSTMs_edge = [
            LSTM(
                'tanh',
                'sigmoid',
                args.lstm_init,
                truncate_gradient=args.truncate_gradient,
                size=args.node_lstm_size,
                rng=rng,
                g_low=-args.g_clip,
                g_high=args.g_clip)
        ]
        edgeRNNs[em] = [
            TemporalInputFeatures(nodeFeatureLength[nm]),
            FCLayer('rectify', args.fc_init, size=args.fc_size, rng=rng),
            FCLayer('linear', args.fc_init, size=args.fc_size, rng=rng),
            multilayerLSTM(
                LSTMs_edge,
                skip_input=True,
                skip_output=True,
                input_output_fused=True)
        ]
        nodeLabels[nm] = T.tensor3(dtype=theano.config.floatX)
    learning_rate = T.scalar(dtype=theano.config.floatX)
    dra = DRA(edgeRNNs,
              nodeRNNs,
              nodeToEdgeConnections,
              edgeListComplete,
              euclidean_loss,
              nodeLabels,
              learning_rate,
              clipnorm=args.clipnorm,
              update_type=gradient_method,
              weight_decay=args.weight_decay)
    return dra


def MaliksRegression(inputDim):
    LSTMs = [
        LSTM(
            'tanh',
            'sigmoid',
            args.lstm_init,
            truncate_gradient=args.truncate_gradient,
            size=args.lstm_size,
            rng=rng,
            g_low=-args.g_clip,
            g_high=args.g_clip), LSTM(
                'tanh',
                'sigmoid',
                args.lstm_init,
                truncate_gradient=args.truncate_gradient,
                size=args.lstm_size,
                rng=rng,
                g_low=-args.g_clip,
                g_high=args.g_clip)
    ]

    layers = [
        TemporalInputFeatures(inputDim),
        # AddNoiseToInput(rng=rng),
        FCLayer('rectify', args.fc_init, size=500, rng=rng),
        FCLayer('linear', args.fc_init, size=500, rng=rng),
        multilayerLSTM(LSTMs, skip_input=True, skip_output=True),
        FCLayer('rectify', args.fc_init, size=500, rng=rng),
        FCLayer('rectify', args.fc_init, size=100, rng=rng),
        FCLayer('linear', args.fc_init, size=inputDim, rng=rng)
    ]

    Y = T.tensor3(dtype=theano.config.floatX)
    learning_rate = T.scalar(dtype=theano.config.floatX)
    rnn = noisyRNN(
        layers,
        euclidean_loss,
        Y,
        learning_rate,
        clipnorm=args.clipnorm,
        update_type=gradient_method,
        weight_decay=args.weight_decay)
    return rnn


def LSTMRegression(inputDim):

    LSTMs = [
        LSTM(
            'tanh',
            'sigmoid',
            args.lstm_init,
            truncate_gradient=args.truncate_gradient,
            size=args.lstm_size,
            rng=rng,
            g_low=-args.g_clip,
            g_high=args.g_clip), LSTM(
                'tanh',
                'sigmoid',
                args.lstm_init,
                truncate_gradient=args.truncate_gradient,
                size=args.lstm_size,
                rng=rng,
                g_low=-args.g_clip,
                g_high=args.g_clip), LSTM(
                    'tanh',
                    'sigmoid',
                    args.lstm_init,
                    truncate_gradient=args.truncate_gradient,
                    size=args.lstm_size,
                    rng=rng,
                    g_low=-args.g_clip,
                    g_high=args.g_clip)
    ]
    layers = [
        TemporalInputFeatures(inputDim),
        multilayerLSTM(LSTMs, skip_input=True, skip_output=True),
        FCLayer('linear', args.fc_init, size=inputDim, rng=rng)
    ]

    Y = T.tensor3(dtype=theano.config.floatX)
    learning_rate = T.scalar(dtype=theano.config.floatX)
    rnn = noisyRNN(
        layers,
        euclidean_loss,
        Y,
        learning_rate,
        clipnorm=args.clipnorm,
        update_type=gradient_method,
        weight_decay=args.weight_decay)
    return rnn


def trainDRA():
    path_to_checkpoint = poseDataset.base_dir + '/{0}/'.format(
        args.checkpoint_path)
    print path_to_checkpoint
    if not os.path.exists(path_to_checkpoint):
        os.mkdir(path_to_checkpoint)
    nodeNames, nodeList, nodeFeatureLength, nodeConnections, edgeList, \
        edgeListComplete, edgeFeatures, nodeToEdgeConnections, trX, trY, \
        trX_validation, trY_validation, trX_forecasting, trY_forecasting, \
        trX_forecast_nodeFeatures = graph.readCRFgraph(poseDataset)

    new_idx = poseDataset.new_idx
    featureRange = poseDataset.nodeFeaturesRanges
    dra = []
    if args.use_pretrained == 1:
        dra = loadDRA(path_to_checkpoint + 'checkpoint.' + str(
            args.iter_to_load))
        print 'DRA model loaded successfully'
    else:
        args.iter_to_load = 0
        if args.dra_type == 'simple':
            dra = DRAmodelRegression(nodeList, edgeList, edgeListComplete,
                                     edgeFeatures, nodeFeatureLength,
                                     nodeToEdgeConnections)
        if args.dra_type == 'RNNatEachNode':
            dra = DRAmodelRegression_RNNatEachNode(
                nodeList, edgeList, edgeListComplete, edgeFeatures,
                nodeFeatureLength, nodeToEdgeConnections)
        if args.dra_type == 'NoEdge':
            dra = DRAmodelRegressionNoEdge(
                nodeList, edgeList, edgeListComplete, edgeFeatures,
                nodeFeatureLength, nodeToEdgeConnections)

    saveForecastedMotion(
        dra.convertToSingleVec(trY_forecasting, new_idx, featureRange),
        path_to_checkpoint)
    saveForecastedMotion(
        dra.convertToSingleVec(trX_forecast_nodeFeatures, new_idx,
                               featureRange), path_to_checkpoint,
        'motionprefix_N_')

    dra.fitModel(
        trX,
        trY,
        snapshot_rate=args.snapshot_rate,
        path=path_to_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        decay_after=args.decay_after,
        learning_rate=args.initial_lr,
        learning_rate_decay=args.learning_rate_decay,
        trX_validation=trX_validation,
        trY_validation=trY_validation,
        trX_forecasting=trX_forecasting,
        trY_forecasting=trY_forecasting,
        trX_forecast_nodeFeatures=trX_forecast_nodeFeatures,
        iter_start=args.iter_to_load,
        decay_type=args.decay_type,
        decay_schedule=args.decay_schedule,
        decay_rate_schedule=args.decay_rate_schedule,
        use_noise=args.use_noise,
        noise_schedule=args.noise_schedule,
        noise_rate_schedule=args.noise_rate_schedule,
        new_idx=new_idx,
        featureRange=featureRange,
        poseDataset=poseDataset,
        graph=graph,
        maxiter=args.maxiter,
        unNormalizeData=unNormalizeData)


def trainMaliks():
    path_to_checkpoint = poseDataset.base_dir + '/{0}/'.format(
        args.checkpoint_path)

    if not os.path.exists(path_to_checkpoint):
        os.mkdir(path_to_checkpoint)

    trX, trY = poseDataset.getMalikFeatures()
    trX_validation, trY_validation = poseDataset.getMalikValidationFeatures()
    trX_forecasting, trY_forecasting \
        = poseDataset.getMalikTrajectoryForecasting()

    saveForecastedMotion(trY_forecasting, path_to_checkpoint)
    saveForecastedMotion(trX_forecasting, path_to_checkpoint,
                         'motionprefix_N_')
    print 'X forecasting ', trX_forecasting.shape
    print 'Y forecasting ', trY_forecasting.shape

    inputDim = trX.shape[2]
    print inputDim
    rnn = []
    if args.use_pretrained == 1:
        rnn = load(path_to_checkpoint + 'checkpoint.' + str(args.iter_to_load))
    else:
        args.iter_to_load = 0
        rnn = MaliksRegression(inputDim)
    rnn.fitModel(
        trX,
        trY,
        snapshot_rate=args.snapshot_rate,
        path=path_to_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        decay_after=args.decay_after,
        learning_rate=args.initial_lr,
        learning_rate_decay=args.learning_rate_decay,
        trX_validation=trX_validation,
        trY_validation=trY_validation,
        trX_forecasting=trX_forecasting,
        trY_forecasting=trY_forecasting,
        iter_start=args.iter_to_load,
        decay_type=args.decay_type,
        decay_schedule=args.decay_schedule,
        decay_rate_schedule=args.decay_rate_schedule,
        use_noise=args.use_noise,
        noise_schedule=args.noise_schedule,
        noise_rate_schedule=args.noise_rate_schedule,
        maxiter=args.maxiter,
        poseDataset=poseDataset,
        unNormalizeData=unNormalizeData)


def trainLSTM():
    path_to_checkpoint = poseDataset.base_dir + '/{0}/'.format(
        args.checkpoint_path)

    if not os.path.exists(path_to_checkpoint):
        os.mkdir(path_to_checkpoint)

    trX, trY = poseDataset.getMalikFeatures()
    trX_validation, trY_validation = poseDataset.getMalikValidationFeatures()
    trX_forecasting, trY_forecasting \
        = poseDataset.getMalikTrajectoryForecasting()

    saveForecastedMotion(trY_forecasting, path_to_checkpoint)
    saveForecastedMotion(trX_forecasting, path_to_checkpoint,
                         'motionprefix_N_')
    print 'X forecasting ', trX_forecasting.shape
    print 'Y forecasting ', trY_forecasting.shape

    inputDim = trX.shape[2]
    print inputDim
    rnn = []
    if args.use_pretrained == 1:
        rnn = load(path_to_checkpoint + 'checkpoint.' + str(args.iter_to_load))
    else:
        args.iter_to_load = 0
        rnn = LSTMRegression(inputDim)
    rnn.fitModel(
        trX,
        trY,
        snapshot_rate=args.snapshot_rate,
        path=path_to_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        decay_after=args.decay_after,
        learning_rate=args.initial_lr,
        learning_rate_decay=args.learning_rate_decay,
        trX_validation=trX_validation,
        trY_validation=trY_validation,
        trX_forecasting=trX_forecasting,
        trY_forecasting=trY_forecasting,
        iter_start=args.iter_to_load,
        decay_type=args.decay_type,
        decay_schedule=args.decay_schedule,
        decay_rate_schedule=args.decay_rate_schedule,
        use_noise=args.use_noise,
        noise_schedule=args.noise_schedule,
        noise_rate_schedule=args.noise_rate_schedule,
        maxiter=args.maxiter,
        poseDataset=poseDataset,
        unNormalizeData=unNormalizeData)


if __name__ == '__main__':

    if args.model_to_train == 'malik':
        trainMaliks()
    elif args.model_to_train == 'dra':
        trainDRA()
    elif args.model_to_train == 'lstm':
        trainLSTM()
    else:
        print "Unknown model type ... existing"
