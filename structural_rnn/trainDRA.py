import sys

try:
    sys.path.remove(
        '/usr/local/lib/python2.7/dist-packages/Theano-0.6.0-py2.7.egg')
except:
    print 'Theano 0.6.0 version not found'

import os
os.environ['PATH'] += ':/usr/local/cuda/bin'
import argparse
import numpy as np
import theano
from theano import tensor as T
from neuralmodels.loadcheckpoint import *
from neuralmodels.costs import softmax_loss, euclidean_loss
from neuralmodels.models import *
from neuralmodels.layers import *
from neuralmodels.updates import Momentum
import cPickle
import readCRFgraph as graph
from unNormalizeData import unNormalizeData
rng = np.random.RandomState(1234567890)

parser = argparse.ArgumentParser()
parser.add_argument('--decay_type', type=str, default='schedule')
parser.add_argument('--decay_after', type=int, default=-1)
parser.add_argument('--initial_lr', type=float, default=1e-3)
parser.add_argument('--learning_rate_decay', type=float, default=1.0)
parser.add_argument('--decay_schedule', nargs='*')
parser.add_argument('--decay_rate_schedule', nargs='*')
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
parser.add_argument('--noise_schedule', nargs='*')
parser.add_argument('--noise_rate_schedule', nargs='*')
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
parser.add_argument('--copy_state', type=int, default=0)
parser.add_argument('--full_skeleton', type=int, default=0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--train_for', type=str, default='validate')
parser.add_argument('--dra_type', type=str, default='simple')
parser.add_argument('--temporal_features', type=int, default=0)
parser.add_argument('--dataset_prefix', type=str, default='')
parser.add_argument('--drop_features', type=int, default=0)
parser.add_argument('--subsample_data', type=int, default=1)
parser.add_argument('--drop_id', type=str, default='')
args = parser.parse_args()

convert_list_to_float = [
    'decay_schedule', 'decay_rate_schedule', 'noise_schedule',
    'noise_rate_schedule'
]
for k in convert_list_to_float:
    if getattr(args, k) is not None:
        temp_list = []
        for v in getattr(args, k):
            temp_list.append(float(v))
        setattr(args, k, temp_list)

print args
if args.use_pretrained:
    print 'Loading pre-trained model with iter={0}'.format(args.iter_to_load)
gradient_method = Momentum(momentum=args.momentum)

drop_ids = args.drop_id.split(',')
drop_id = []
for dids in drop_ids:
    drop_id.append(int(dids))
'''Loads H3.6m dataset'''
sys.path.insert(0, 'CRFProblems/H3.6m')
import processdata as poseDataset
poseDataset.T = args.sequence_length
poseDataset.delta_shift = args.sequence_length - args.sequence_overlap
poseDataset.num_forecast_examples = 24
poseDataset.copy_state = args.copy_state
poseDataset.full_skeleton = args.full_skeleton
poseDataset.train_for = args.train_for
poseDataset.temporal_features = args.temporal_features
poseDataset.crf_file = './CRFProblems/H3.6m/crf' + args.crf
poseDataset.dataset_prefix = args.dataset_prefix
poseDataset.drop_features = args.drop_features
poseDataset.drop_id = drop_id
poseDataset.subsample_data = args.subsample_data
poseDataset.runall()
if poseDataset.copy_state:
    args.batch_size = poseDataset.minibatch_size

print '**** H3.6m Loaded ****'


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


def DRAmodelClassification(nodeList,
                           edgeList,
                           edgeListComplete,
                           edgeFeatures,
                           nodeFeatureLength,
                           nodeToEdgeConnections,
                           clipnorm=0.0):
    '''
    Understanding data structures
    nodeToEdgeConnections: node_type ---> [edge_types]
    nodeConnections: node_name ---> [node_names]
    nodeNames: node_name ---> node_type
    nodeList: node_type ---> output_dimension
    nodeFeatureLength: node_type ---> feature_dim_into_nodeRNN

    edgeList: list of edge types
    edgeFeatures: edge_type ---> feature_dim_into_edgeRNN
    '''

    edgeRNNs = {}
    edgeNames = edgeList
    lstm_init = 'orthogonal'
    softmax_init = 'uniform'

    for em in edgeNames:
        inputJointFeatures = edgeFeatures[em]
        edgeRNNs[em] = [
            TemporalInputFeatures(inputJointFeatures),
            AddNoiseToInput(rng=rng), simpleRNN(
                'rectify',
                'uniform',
                size=500,
                temporal_connection=False,
                rng=rng), simpleRNN(
                    'rectify',
                    'uniform',
                    size=500,
                    temporal_connection=False,
                    rng=rng),
            LSTM('tanh', 'sigmoid', lstm_init, 100, 1000, rng=rng),
            LSTM('tanh', 'sigmoid', lstm_init, 100, 1000, rng=rng)
        ]

    nodeRNNs = {}
    nodeNames = nodeList.keys()
    nodeLabels = {}
    for nm in nodeNames:
        num_classes = nodeList[nm]
        nodeRNNs[nm] = [
            LSTM('tanh', 'sigmoid', lstm_init, 100, 1000, rng=rng), simpleRNN(
                'rectify',
                'uniform',
                size=500,
                temporal_connection=False,
                rng=rng), simpleRNN(
                    'rectify',
                    'uniform',
                    size=100,
                    temporal_connection=False,
                    rng=rng), simpleRNN(
                        'rectify',
                        'uniform',
                        size=54,
                        temporal_connection=False,
                        rng=rng), softmax(num_classes, softmax_init, rng=rng)
        ]
        em = nm + '_input'
        edgeRNNs[em] = [
            TemporalInputFeatures(nodeFeatureLength[nm]),
            AddNoiseToInput(rng=rng),
            simpleRNN(
                'rectify',
                'uniform',
                size=500,
                temporal_connection=False,
                rng=rng),
            simpleRNN(
                'rectify',
                'uniform',
                size=500,
                temporal_connection=False,
                rng=rng),
        ]
        nodeLabels[nm] = T.lmatrix()
    learning_rate = T.fscalar()
    dra = DRA(edgeRNNs,
              nodeRNNs,
              nodeToEdgeConnections,
              edgeListComplete,
              softmax_loss,
              nodeLabels,
              learning_rate,
              clipnorm=clipnorm)
    return dra


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
            #AddNoiseToInput(rng=rng),
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
            #AddNoiseToInput(rng=rng),
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
            #LSTM('tanh','sigmoid',args.lstm_init,truncate_gradient=args.truncate_gradient,size=args.lstm_size,rng=rng,g_low=-args.g_clip,g_high=args.g_clip)
        ]
        edgeRNNs[em] = [
            TemporalInputFeatures(inputJointFeatures),
            #AddNoiseToInput(rng=rng),
            FCLayer('rectify', args.fc_init, size=args.fc_size, rng=rng),
            FCLayer('linear', args.fc_init, size=args.fc_size, rng=rng)
            #multilayerLSTM(LSTMs,skip_input=True,skip_output=True,input_output_fused=True)
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
            #AddNoiseToInput(rng=rng),
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
        #AddNoiseToInput(rng=rng),
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
    saveNormalizationStats(path_to_checkpoint)
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
    saveNormalizationStats(path_to_checkpoint)

    trX, trY = poseDataset.getMalikFeatures()
    trX_validation, trY_validation = poseDataset.getMalikValidationFeatures()
    trX_forecasting, trY_forecasting = poseDataset.getMalikTrajectoryForecasting(
    )

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
    saveNormalizationStats(path_to_checkpoint)

    trX, trY = poseDataset.getMalikFeatures()
    trX_validation, trY_validation = poseDataset.getMalikValidationFeatures()
    trX_forecasting, trY_forecasting = poseDataset.getMalikTrajectoryForecasting(
    )

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


def saveNormalizationStats(path):
    activities = {}
    activities['walking'] = 14
    activities['eating'] = 4
    activities['smoking'] = 11
    activities['discussion'] = 3
    activities['walkingdog'] = 15

    cPickle.dump(poseDataset.data_stats,
                 open('{0}h36mstats.pik'.format(path), 'wb'))
    forecastidx = poseDataset.data_stats['forecastidx']
    num_forecast_examples = len(forecastidx.keys())
    f = open('{0}forecastidx'.format(path), 'w')
    for i in range(num_forecast_examples):
        tupl = forecastidx[i]
        st = '{0},{1},{2},{3}\n'.format(i, activities[tupl[0]], tupl[2],
                                        tupl[1])
        f.write(st)
    f.close()
    print "************Done saving the stats*********"


if __name__ == '__main__':

    if args.model_to_train == 'malik':
        trainMaliks()
    elif args.model_to_train == 'dra':
        trainDRA()
    elif args.model_to_train == 'lstm':
        trainLSTM()
    else:
        print "Unknown model type ... existing"
