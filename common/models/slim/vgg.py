from ..slim import *
from .nets import vgg

class Vgg16(SlimNet):

    def __init__(self, inputs, keep_prob=1.0, base_trainable=False, is_training=False,
                 weight_decay=0.0005):
        super(Vgg16, self).__init__(inputs, keep_prob, base_trainable, is_training)
        self._prelogits_names = ["vgg_16/fc8"]
        self._scope_name = 'vgg_16'
        self._weight_decay = weight_decay
        self.net, self.end_points = self._build_net()

    def _build_net(self):
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=self._weight_decay)):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], trainable=self._cnn_trainable):

                with tf.variable_scope(self.scope_name, 'vgg_16') as scope:
                    net, end_points = vgg.vgg_16(self._inputs, num_classes=None, is_training=self._cnn_trainable,
                                                 dropout_keep_prob=self._keep_prob, scope=scope)
        return net, end_points

    def pre_logits(self):
        return slim.flatten(self.net, scope='logits')

    def output_logits(self, num_labels, net=None, scope='fc8', reuse=tf.AUTO_REUSE):

        net = self.net if not net else net

        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=self._weight_decay)):
            with tf.variable_scope(self.scope_name, 'vgg_16', reuse=reuse):

                logits = slim.dropout(net, self._keep_prob, is_training=self._is_training,
                                      scope='dropout7')
                logits = slim.conv2d(logits, num_labels, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope=scope)
                logits = slim.flatten(logits, scope="flatten")
        return logits