import os
import argparse
import json

import tensorflow as tf
import numpy as np

import gnn
from gnn.data import load_data
from gnn.utils import gumbel_softmax


def model_fn(features, labels, mode, params):
    pred_stack = gnn.dynamical.dynamical_multisteps(features,
                                                    params,
                                                    params['pred_steps'],
                                                    training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {'next_steps': pred_stack}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    n_conv_layers = len(params['cnn']['filters'])
    expected_time_series = gnn.utils.stack_time_series(features['time_series'][:, 2*n_conv_layers+1:, :, :],
                                                       params['pred_steps'])

    loss = tf.losses.mean_squared_error(expected_time_series,
                                        pred_stack[:, :-params['pred_steps'], :, :, :])

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            learning_rate=params['learning_rate'],
            global_step=tf.train.get_global_step(),
            decay_steps=100,
            decay_rate=0.95,
            staircase=True,
            name='learning_rate'
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Use the loss between adjacent steps in original time_series as baseline
    time_series_loss_baseline = tf.metrics.mean_squared_error(features['time_series'][:, 1:, :, :],
                                                              features['time_series'][:, :-1, :, :])

    eval_metric_ops = {'time_series_loss_baseline': time_series_loss_baseline}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    with open(ARGS.config) as f:
        model_params = json.load(f)

    model_params['pred_steps'] = ARGS.pred_steps

    cnn_multistep_regressor = tf.estimator.Estimator(
        model_fn=model_fn,
        params=model_params,
        model_dir=ARGS.log_dir)

    if ARGS.train:
        if model_params.get('edge_types', 0) > 1:
            train_data, train_edge = load_data(ARGS.data_dir, ARGS.data_transpose, edge=True,
                                               prefix='train')
            train_edge = gnn.utils.one_hot(train_edge, model_params['edge_types'], np.float32)

            features = {'time_series': train_data, 'edge_type': train_edge}
        else:
            train_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                                   prefix='train')
            features = {'time_series': train_data}

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=features,
            batch_size=ARGS.batch_size,
            num_epochs=None,
            shuffle=True
        )

        cnn_multistep_regressor.train(input_fn=train_input_fn,
                                      steps=ARGS.train_steps)

    # Evaluation
    if ARGS.eval:
        if model_params.get('edge_types', 0) > 1:
            valid_data, valid_edge = load_data(ARGS.data_dir, ARGS.data_transpose, edge=True,
                                               prefix='valid')
            valid_edge = gnn.utils.one_hot(valid_edge, model_params['edge_types'], np.float32)

            features = {'time_series': valid_data, 'edge_type': valid_edge}
        else:
            valid_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                                   prefix='valid')
            features = {'time_series': valid_data}

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=features,
            batch_size=ARGS.batch_size,
            num_epochs=1,
            shuffle=False
        )
        eval_results = cnn_multistep_regressor.evaluate(input_fn=eval_input_fn)

        if not ARGS.verbose:
            print('Evaluation results: {}'.format(eval_results))

    # Prediction
    if ARGS.test:
        if model_params.get('edge_types', 0) > 1:
            test_data, test_edge = load_data(ARGS.data_dir, ARGS.data_transpose, edge=True,
                                             prefix='test')
            test_edge = gnn.utils.one_hot(test_edge, model_params['edge_types'], np.float32)

            features = {'time_series': test_data, 'edge_type': test_edge}
        else:
            test_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                                  prefix='test')
            features = {'time_series': test_data}

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=features,
            batch_size=ARGS.batch_size,
            shuffle=False
        )
        prediction = cnn_multistep_regressor.predict(input_fn=predict_input_fn)
        prediction = np.array([pred['next_steps'] for pred in prediction])
        np.save(os.path.join(ARGS.log_dir, 'prediction_{}.npy'.format(
            ARGS.pred_steps)), prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    parser.add_argument('--data-transpose', type=int, nargs=4, default=None,
                        help='axes for data transposition')
    parser.add_argument('--config', type=str,
                        help='model config file')
    parser.add_argument('--log-dir', type=str,
                        help='log directory')
    parser.add_argument('--train-steps', type=int, default=1000,
                        help='number of training steps')
    parser.add_argument('--pred-steps', type=int, default=1,
                        help='number of steps the estimator predicts for time series')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='turn on logging info')
    parser.add_argument('--train', action='store_true', default=False,
                        help='turn on training')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='turn on evaluation')
    parser.add_argument('--test', action='store_true', default=False,
                        help='turn on test')
    ARGS = parser.parse_args()

    if ARGS.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    main()
