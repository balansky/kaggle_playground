from tensorflow.contrib import slim
import tensorflow as tf


class SlimNet(object):

    def __init__(self, inputs, keep_prob=1.0, base_trainable=False, is_training=False):
        self._scope_name = None
        self._prelogits_names = []
        self._inputs = inputs
        self._keep_prob = keep_prob
        self._cnn_trainable = base_trainable
        self._is_training = is_training

    @property
    def prelogits_names(self):
        return self._prelogits_names

    @prelogits_names.setter
    def prelogits_names(self, value):
        self._prelogits_names = value

    @property
    def scope_name(self):
        return self._scope_name

    def _build_net(self):
        raise NotImplementedError()

    def variables_without_prelogits(self):
        scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_name)
        if self._prelogits_names:
            variables_to_restore = []
            for var in scope_vars:
                excluded = False
                for exclusion in self._prelogits_names:
                    if var.op.name.startswith(exclusion):
                        excluded = True
                        break
                if not excluded:
                    variables_to_restore.append(var)
        else:
            variables_to_restore = scope_vars
        return variables_to_restore

    def restore_fn(self, ckpt_dir, pretrained_ckpt=None):
        try:
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
            if not ckpt_path and pretrained_ckpt:
                ckpt_path = pretrained_ckpt
                restore_variables = self.variables_without_prelogits()
            else:
                restore_variables = tf.trainable_variables()
            return slim.assign_from_checkpoint_fn(ckpt_path, restore_variables, ignore_missing_vars=False)
        except Exception as err:
            tf.logging.warning(err)
            return None