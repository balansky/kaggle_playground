# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib import slim


def transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.
    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)
    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output


def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
    """Batch Spatial Transformer Layer
    Parameters
    ----------
    U : float
        tensor of inputs [num_batch,height,width,num_channels]
    thetas : float
        a set of transformations for each input [num_batch,num_transforms,6]
    out_size : int
        the size of the output [out_height,out_width]
    Returns: float
        Tensor of size [num_batch*num_transforms,out_height,out_width,num_channels]
    """
    with tf.variable_scope(name):
        num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i]*num_transforms for i in xrange(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        return transformer(input_repeated, thetas, out_size)


def conv2d_block(inp, scale, *args, **kwargs):
    return inp + slim.conv2d(inp, *args, **kwargs) * scale

def localisation_net(inputs, keep_prob, is_training):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'scale': True
    }
    with tf.variable_scope('Spatial_Transformer'):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(0.00004),
                                biases_regularizer=slim.l2_regularizer(0.00004),
                                weights_initializer=tf.truncated_normal_initializer(),
                                biases_initializer=tf.truncated_normal_initializer(),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                activation_fn=tf.nn.relu
                                ):
                net = slim.conv2d(inputs, 32, [7, 7], 4, scope='Conv2d_1a')
                # net = slim.repeat(net, 3, conv2d_block, 0.1, 32, [4, 4], 2,
                #                   scope='Conv2d_1b_a')
                # net = slim.max_pool2d(net, [2, 2], scope='Pool_1')

                # net = slim.conv2d(net, 32, [4, 4], 2, scope='Conv2d_1a_b')
                # net = slim.repeat(net, 3, conv2d_block, 0.1, 32, [4, 4], 1,
                #                   scope='Conv2d_1b_b')

                net = slim.conv2d(net, 32, [4, 4], 2, scope='Conv2d_2a')
                # net = slim.repeat(net, 3, conv2d_block, 0.1, 64, [4, 4], 1,
                #                   scope='Conv2d_2b')

                # net = slim.max_pool2d(net, [2, 2], scope='Pool_2')

                net = slim.conv2d(net, 64, [4, 4], 2, scope='Conv2d_3a')
                # net = slim.repeat(net, 3, conv2d_block, 0.1, 64, [4, 4], 1,
                #                   scope='Conv2d_2b_b')

                net = slim.conv2d(net, 64, [4, 4], 2, scope='Conv2d_4a')
                # net = slim.repeat(net, 3, conv2d_block, 0.1, 128, [4, 4], 1,
                #                   scope='Conv2d_3b')

                # net = slim.max_pool2d(net, [2, 2], scope='Pool_3')

                # net = slim.conv2d(net, 128, [4, 4], 2, scope='Conv2d_5a')
                # net = slim.repeat(net, 3, conv2d_block, 0.1, 256, [4, 4], 1,
                #                   scope='Conv2d_4b')
                # net = slim.max_pool2d(net, [2, 2], scope='Pool_4')

                net = slim.conv2d(net, 20, net.get_shape()[1:3], padding='Valid', scope='Full_1')

                net = slim.dropout(net, keep_prob, scope='dropout_1')

                # net = slim.conv2d(net, 32, [1, 1], padding='Valid', scope='Full_2')
                #
                # net = slim.dropout(net, keep_prob, scope='dropout_2')

                net = slim.flatten(net)
    return net



# def localisation_net_448(inputs, keep_prob, is_training):
#     # batch_norm_params = {
#     #     # Decay for the moving averages.
#     #     'decay': 0.9997,
#     #     # epsilon to prevent 0s in variance.
#     #     'epsilon': 0.001,
#     #     # collection containing update_ops.
#     #     'updates_collections': tf.GraphKeys.UPDATE_OPS,
#     #     'scale': True
#     # }
#     with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                         weights_regularizer=slim.l2_regularizer(0.00004),
#                         biases_regularizer=slim.l2_regularizer(0.00004),
#                         # weights_initializer=tf.truncated_normal_initializer(),
#                         # biases_initializer=tf.truncated_normal_initializer(),
#                         activation_fn=tf.nn.relu
#                         ):
#         with tf.variable_scope('Spatial_Transformer') as sc:
#
#             with slim.arg_scope([slim.dropout], is_training=is_training):
#                 net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv0')
#                 net = slim.max_pool2d(net, [2, 2], scope='pool0')
#                 net = slim.repeat(net, 1, slim.conv2d, 64, [3, 3], scope='conv1')
#                 net = slim.max_pool2d(net, [2, 2], scope='pool1')
#                 net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
#                 net = slim.max_pool2d(net, [2, 2], scope='pool2')
#                 net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
#                 net = slim.max_pool2d(net, [2, 2], scope='pool3')
#                 net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
#                 net = slim.max_pool2d(net, [2, 2], scope='pool4')
#                 net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
#                 net = slim.max_pool2d(net, [2, 2], scope='pool5')
#                 net = slim.conv2d(net, 20, [7, 7], padding='VALID', scope='Full_1')
#
#                 net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout6')
#
#                 net = slim.conv2d(net, 32, [1, 1], padding='VALID', scope='Full_2')
#
#                 net = slim.dropout(net, keep_prob, scope='dropout7')
#
#                 net = slim.flatten(net, scope='Flatten')
#                 # w_fc1 = tf.Variable(tf.zeros([32, 6]), name='W_fc1')
#                 # b_fc1 = tf.Variable(initial_value=[1.0, 0, 0, 0, 1.0, 0], dtype=tf.float32, name='b_fc1')
#                 # h_fc1 = tf.matmul(net, w_fc1) + b_fc1
#
#                 #     with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
#                 #                         stride=1, padding='VALID'):
#                 #         net = slim.conv2d(inputs, 20, [5, 5], stride=1,
#                 #                           scope='Conv2d_1a_5x5')
#                 #         net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID',
#                 #                               scope='MaxPool_1a_3x3')
#                 #         net = slim.conv2d(net, 20, [5, 5], stride=1,
#                 #                           scope='Conv2d_1b_5x5')
#                 #         net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID',
#                 #                               scope='MaxPool_1b_3x3')
#                 #         # net = slim.conv2d(net, 32, [5, 5], stride=1,
#                 #         #                   padding='SAME', scope='Conv2d_2a_5x5')
#                 #         # net = slim.max_pool2d(net, [5, 5], stride=2, padding='VALID',
#                 #         #                       scope='MaxPool_2a_3x3')
#                 #         # net = slim.conv2d(net, 32, [5, 5], stride=1,
#                 #         #                   padding='SAME', scope='Conv2d_2b_5x5')
#                 #         # net = slim.max_pool2d(net, [5, 5], stride=2, padding='VALID',
#                 #         #                       scope='MaxPool_2b_3x3')
#                 #         net = slim.conv2d(net, 20, net.get_shape()[1:3], padding='VALID', scope='Full_1')
#                 #         slim.dropout(net, keep_prob, is_training=is_training,
#                 #                      scope='Dropout_1')
#                 #         net = slim.conv2d(net, 32, [1, 1], padding='VALID', scope='Full_2')
#                 #         slim.dropout(net, keep_prob, is_training=is_training,
#                 #                      scope='Dropout_2')
#                 # net = slim.flatten(net, scope='Flatten')
#                 # w_fc1 = tf.Variable(tf.zeros([32, 6]), name='W_fc1')
#                 # b_fc1 = tf.Variable(initial_value=[1.0, 0, 0, 0, 1.0, 0], dtype=tf.float32, name='b_fc1')
#                 # h_fc1 = tf.matmul(net, w_fc1) + b_fc1
#
#     return net

def hfc(net,output_size=1):
    initial_value = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] * output_size
    w_fc1 = tf.Variable(tf.zeros([20, output_size*6]), name='Spatial_Transformer/W_fc1')
    b_fc1 = tf.Variable(initial_value=initial_value, dtype=tf.float32, name='Spatial_Transformer/b_fc1')
    # h_fc1 = tf.tanh(tf.matmul(net, w_fc1) + b_fc1)
    h_fc1 = tf.matmul(net, w_fc1) + b_fc1
    return h_fc1


def transform_448(inputs, keep_prob, is_training, output_width, output_height, output_size=1):
    net = localisation_net(inputs, keep_prob, is_training)
    h_fc1 = hfc(net, output_size)
    if output_size == 1:
        h_trans = transformer(inputs, h_fc1, (output_width, output_height))
    else:
        split_h_fc = tf.split(h_fc1, output_size, 1)
        stk_h_fc = tf.stack(split_h_fc, 1)
        h_trans = batch_transformer(inputs, stk_h_fc, (output_width, output_height))
    return h_trans

