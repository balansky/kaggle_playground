from ..slim import *
from tensorflow.contrib.slim.python.slim.nets import inception_v3
from .nets import inception_v4, inception_resnet_v2


class InceptV3(SlimNet):

    def __init__(self, inputs, keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, stddev=0.1):
        super(InceptV3, self).__init__(inputs, keep_prob, base_trainable, is_training)
        self._prelogits_names = ["InceptionV3/Logits", "InceptionV3/AuxLogits"]
        self._scope_name = 'InceptionV3'
        self._weight_decay = weight_decay
        self._stddev = stddev
        self.net, self.end_points = self._build_net()

    def _build_net(self):

        batch_norm_params = {
            "is_training": self._cnn_trainable,
            "trainable": self._cnn_trainable,
            # Decay for the moving averages.
            "decay": 0.9997,
            # Epsilon to prevent 0s in variance.
            "epsilon": 0.001,
            # Collection containing the moving mean and moving variance.
            "variables_collections": {
                "beta": None,
                "gamma": None,
                "moving_mean": ["moving_vars"],
                "moving_variance": ["moving_vars"],
            }
        }
        with slim.arg_scope(inception_v3.inception_v3_arg_scope(weight_decay=self._weight_decay, stddev=self._stddev)):

            with tf.variable_scope(self.scope_name, 'InceptionV3', [self._inputs]) as scope:
                with slim.arg_scope(
                        [slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
                    with slim.arg_scope([slim.conv2d, slim.fully_connected], trainable=self._cnn_trainable):
                        net, end_points = inception_v3.inception_v3_base(self._inputs, scope=scope)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a')
        return net, end_points

    def pre_logits(self):
        return slim.flatten(self.net, scope='logits')

    def output_logits(self, num_labels, net=None, scope='Logits', reuse=tf.AUTO_REUSE):

        net = net if net is not None else self.net

        with slim.arg_scope(inception_v3.inception_v3_arg_scope(weight_decay=self._weight_decay, stddev=self._stddev)):
            with tf.variable_scope(self.scope_name, 'InceptionV3', reuse=reuse):
                with tf.variable_scope(scope):

                    logits = slim.dropout(net, keep_prob=self._keep_prob, is_training=self._is_training,
                                          scope='Dropout_1b')
                    logits = slim.conv2d(logits, num_labels, [1, 1], activation_fn=None,
                                         normalizer_fn=None, scope='Conv2d_1c_1x1')
                    logits = slim.flatten(logits, scope="flatten")
        return logits


class InceptV4(SlimNet):

    def __init__(self, inputs, keep_prob=1.0, base_trainable=False, is_training=False,
                 use_batch_norm=True, weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        super(InceptV4, self).__init__(inputs, keep_prob, base_trainable, is_training)
        self._prelogits_names = ["InceptionV4/Logits", "InceptionV4/AuxLogits"]
        self._scope_name = 'InceptionV4'
        self._use_batch_norm = use_batch_norm
        self._batch_norm_decay = batch_norm_decay
        self._weight_decay = weight_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self.net, self.end_points = self._build_net()

    def _build_net(self):

        with slim.arg_scope(inception_v4.inception_v4_arg_scope(use_batch_norm=self._use_batch_norm,
                                                                weight_decay=self._weight_decay,
                                                                batch_norm_decay=self._batch_norm_decay,
                                                                batch_norm_epsilon=self._batch_norm_epsilon)):
            with tf.variable_scope(self.scope_name, 'InceptionV4') as scope:
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.batch_norm], trainable=self._cnn_trainable):
                    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self._cnn_trainable):
                        net, end_points = inception_v4.inception_v4_base(self._inputs, scope=scope)
                        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a')
        return net, end_points

    def pre_logits(self):
        return slim.flatten(self.net, scope='logits')

    def output_logits(self, num_labels, net=None, scope='Logits', reuse=tf.AUTO_REUSE):

        net = net if net is not None else self.net

        with slim.arg_scope(inception_v4.inception_v4_arg_scope(use_batch_norm=self._use_batch_norm,
                                                                weight_decay=self._weight_decay,
                                                                batch_norm_decay=self._batch_norm_decay,
                                                                batch_norm_epsilon=self._batch_norm_epsilon)):
            with tf.variable_scope(self.scope_name, 'InceptionV4', reuse=reuse):
                with tf.variable_scope(scope):
                    logits = slim.dropout(net, keep_prob=self._keep_prob, is_training=self._is_training,
                                       scope='Dropout_1b')
                    logits = slim.flatten(logits, scope='PreLogitsFlatten')
                    logits = slim.fully_connected(logits, num_labels, activation_fn=None,
                                                  normalizer_fn=None, scope='Logits')
        return logits


class InceptResV2(SlimNet):

    def __init__(self, inputs, keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        super(InceptResV2, self).__init__(inputs, keep_prob, base_trainable, is_training)
        self._prelogits_names = ["InceptionResnetV2/AuxLogits", "InceptionResnetV2/Logits"]
        self._scope_name = 'InceptionResnetV2'
        self._batch_norm_decay = batch_norm_decay
        self._weight_decay = weight_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self.net, self.end_points = self._build_net()

    def _build_net(self):

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(
                weight_decay=self._weight_decay,
                batch_norm_decay=self._batch_norm_decay,
                batch_norm_epsilon=self._batch_norm_epsilon)):
            with tf.variable_scope(self.scope_name, 'InceptionResnetV2') as scope:
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.batch_norm], trainable=self._cnn_trainable):
                    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self._cnn_trainable):
                        net, end_points = inception_resnet_v2.inception_resnet_v2_base(self._inputs, scope=scope)
                        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a')
        return net, end_points

    def pre_logits(self):
        return slim.flatten(self.net, scope='logits')

    def output_logits(self, num_labels, net=None, scope='Logits', reuse=tf.AUTO_REUSE):

        net = net if net is not None else self.net

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(
                weight_decay=self._weight_decay,
                batch_norm_decay=self._batch_norm_decay,
                batch_norm_epsilon=self._batch_norm_epsilon)):

            with tf.variable_scope(self.scope_name, 'InceptionResnetV2', reuse=reuse):
                with tf.variable_scope(scope):
                    logits = slim.dropout(net, keep_prob=self._keep_prob, is_training=self._is_training,
                                       scope='Dropout_1b')
                    logits = slim.flatten(logits, scope='PreLogitsFlatten')
                    logits = slim.fully_connected(logits, num_labels, activation_fn=None,
                                                  normalizer_fn=None, scope='Logits')
        return logits