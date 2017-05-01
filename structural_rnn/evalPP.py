#!/usr/bin/env python2
"""Forecasts poses in the format expected by my stats calculation code."""

import argparse
import json
import os
import sys

import h5py
import numpy as np
from neuralmodels.loadcheckpoint import loadDRA
import readCRFgraph as graph

import crfproblems.stdpp.processdata as poseDataset


def f32(x):
    return np.asarray(x, dtype='float32')


def write_baseline(dest_dir, dataset, steps_to_predict, method):
    meth_name = method.method_name
    out_path = os.path.join(dest_dir, 'results_' + meth_name + '.h5')
    print('Writing %s baseline to %s' % (meth_name, out_path))

    extra_data = {}
    if dataset.is_3d:
        cond_on, pred_on = dataset.get_ds_for_eval(train=False)
        pred_on_orig = f32(dataset.reconstruct_skeletons(pred_on))
        pred_usable = pred_scales = None
    else:
        evds = dataset.get_ds_for_eval(train=False)
        if dataset.has_sparse_annos:
            cond_on, pred_on, pred_scales, pred_usable = evds
        else:
            cond_on, pred_on, pred_scales = evds
            pred_usable = None
        extra_data['pck_joints'] = dataset.pck_joints
        pred_on_orig = f32(dataset.reconstruct_poses(pred_on))

    if pred_usable is None:
        pred_usable = np.ones(pred_on_orig.shape[:2], dtype=bool)

    if pred_scales is None:
        pred_scales = np.ones(pred_on_orig.shape[:2], dtype='float32')

    result = method(cond_on, steps_to_predict)
    if dataset.is_3d:
        result = f32(dataset.reconstruct_skeletons(result))
    else:
        result = f32(dataset.reconstruct_poses(result))
    # insert an extra axis
    result = result[:, None]
    assert (result.shape[0],) + result.shape[2:] == pred_on_orig.shape, \
        (result.shape, pred_on_orig.shape)
    with h5py.File(out_path, 'w') as fp:
        fp['/method_name'] = meth_name
        if dataset.is_3d:
            fp['/parents_3d'] = dataset.parents
            fp.create_dataset(
                '/skeletons_3d_true',
                compression='gzip',
                shuffle=True,
                data=pred_on_orig)
            fp.create_dataset(
                '/skeletons_3d_pred',
                compression='gzip',
                shuffle=True,
                data=result)
        else:
            fp['/parents_2d'] = dataset.parents
            fp.create_dataset(
                '/poses_2d_true',
                compression='gzip',
                shuffle=True,
                data=pred_on_orig)
            fp['/scales_2d'] = f32(pred_scales)
            fp.create_dataset(
                '/poses_2d_pred',
                compression='gzip',
                shuffle=True,
                data=result)
        fp['/is_usable'] = pred_usable
        fp['/extra_data'] = json.dumps(extra_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint',
        help='checkpoint to load from')
    parser.add_argument('--forecast', default='srnn')
    parser.add_argument('--train_for', default='final')
    parser.add_argument(
        '--is-3d',
        action='store_true',
        default=False,
        help='read 3D data from dataset, rather than 2D data')
    parser.add_argument('dataset_path', help='path to .h5 containing data')
    parser.add_argument(
        'output_path', help='where to write generated poses to')
    args = parser.parse_args()

    # poseDataset.T = 150
    # poseDataset.delta_shift = 100
    # poseDataset.num_forecast_examples = 24
    # poseDataset.motion_prefix = args.motion_prefix
    # poseDataset.motion_suffix = args.motion_suffix
    # poseDataset.temporal_features = args.temporal_features
    # poseDataset.full_skeleton = args.full_skeleton
    # poseDataset.dataset_prefix = args.dataset_prefix
    # poseDataset.crf_file = './crfproblems/h36m/crf'
    # poseDataset.train_for = args.train_for
    # poseDataset.drop_features = args.drop_features
    # poseDataset.drop_id = [args.drop_id]
    # poseDataset.runall()
    # Loads H3.6m dataset
    poseDataset.T = 150
    poseDataset.train_for = args.train_for
    poseDataset.crf_file = './crfproblems/stdpp/crf'
    poseDataset.ds_label = 'ikea'  # XXX: this should change for other DS!
    poseDataset.ds_path = args.dataset_path
    poseDataset.ds_is_3d = args.is_3d
    poseDataset.runall()

    new_idx = poseDataset.new_idx
    featureRange = poseDataset.nodeFeaturesRanges
    path_to_checkpoint = args.checkpoint

    if args.forecast == 'srnn':
        print "Using checkpoint at: ", path_to_checkpoint
        assert os.path.exists(path_to_checkpoint), \
            "%s doesn't exist" % path_to_checkpoint

        print 'Loading the model (this takes long, can take upto 25 minutes)'
        model = loadDRA(path_to_checkpoint)
        print 'Loaded S-RNN from ', path_to_checkpoint

        # next call reqd. to init variables in poseDataset module
        graph.readCRFgraph(poseDataset)

        def method(cond_on, steps_to_predict):
            # assumes cond_on is N*T*D
            trX_forecasting, trX_forecast_nodeFeatures \
                = graph.convert_forecast_data(poseDataset, cond_on)
            forecasted_motion = model.predict_sequence(
                trX_forecasting,
                trX_forecast_nodeFeatures,
                sequence_length=steps_to_predict,
                poseDataset=poseDataset,
                graph=graph)
            fcst_tnd = model.convertToSingleVec(forecasted_motion, new_idx,
                                                featureRange)
            fcst_ntd = fcst_tnd.transpose((1, 0, 2))
            return fcst_ntd

        method.method_name = 'srnn'
    # elif args.forecast == 'lstm3lr' or args.forecast == 'erd':
    #     path_to_checkpoint = '{0}checkpoint.{1}'.format(path, iteration)
    #     if os.path.exists(path_to_checkpoint):
    #         print "Loading the model {0} (this may take sometime)".format(
    #             args.forecast)
    #         model = load(path_to_checkpoint)
    #         print 'Loaded the model from ', path_to_checkpoint

    #         trX_forecasting, trY_forecasting = poseDataset.getMalikTrajectoryForecasting()

    #         fname = 'ground_truth_forecast'
    #         model.saveForecastedMotion(trY_forecasting, path, fname)

    #         fname = 'motionprefix'
    #         model.saveForecastedMotion(trX_forecasting, path, fname)

    #         forecasted_motion = model.predict_sequence(
    #             trX_forecasting, sequence_length=trY_forecasting.shape[0])
    #         fname = 'forecast'
    #         model.saveForecastedMotion(forecasted_motion, path, fname)

    #         skel_err = np.mean(
    #             np.sqrt(
    #                 np.sum(
    #                     np.square((forecasted_motion - trY_forecasting)), axis=2)),
    #             axis=1)
    #         err_per_dof = skel_err / trY_forecasting.shape[2]
    #         fname = 'forecast_error'
    #         model.saveForecastError(skel_err, err_per_dof, path, fname)

    #         del model

    # elif args.forecast == 'dracell':
    #     path_to_checkpoint = '{0}checkpoint.{1}'.format(path, iteration)
    #     if os.path.exists(path_to_checkpoint):
    #         [
    #             nodeNames, nodeList, nodeFeatureLength, nodeConnections, edgeList,
    #             edgeListComplete, edgeFeatures, nodeToEdgeConnections, trX, trY,
    #             trX_validation, trY_validation, trX_forecasting, trY_forecasting,
    #             trX_forecast_nodeFeatures
    #         ] = graph.readCRFgraph(
    #             poseDataset, noise=0.7, forecast_on_noisy_features=True)
    #         print trX_forecast_nodeFeatures.keys()
    #         print 'Loading the model'
    #         model = loadDRA(path_to_checkpoint)
    #         print 'Loaded DRA: ', path_to_checkpoint
    #         t0 = time.time()
    #         trY_forecasting = model.convertToSingleVec(trY_forecasting, new_idx,
    #                                                    featureRange)

    #         trX_forecast_nodeFeatures_ = model.convertToSingleVec(
    #             trX_forecast_nodeFeatures, new_idx, featureRange)
    #         fname = 'motionprefixlong'
    #         model.saveForecastedMotion(trX_forecast_nodeFeatures_, path, fname)

    #         cellstate = model.predict_cell(
    #             trX_forecasting,
    #             trX_forecast_nodeFeatures,
    #             sequence_length=trY_forecasting.shape[0],
    #             poseDataset=poseDataset,
    #             graph=graph)
    #         fname = 'forecast_celllong_{0}'.format(iteration)
    #         model.saveCellState(cellstate, path, fname)
    #         t1 = time.time()
    #         del model


if __name__ == '__main__':
    main()
