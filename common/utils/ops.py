import tensorflow as tf
from tensorflow.contrib import slim


def lr_decay_op(decay_frequency=None, decay_rate=None):
    def _lr_decay_fn(init_learning_rate, gs):

        if decay_frequency and decay_rate:
            return tf.train.exponential_decay(init_learning_rate, gs, decay_frequency,
                                              decay_rate, staircase=True)
        else:
            return None
    return _lr_decay_fn


def train_op(total_loss, learning_rate, optimizer, decay_frequency=None, decay_rate=None, clip_gradients=None):

    _lr_decay_fn = lr_decay_op(decay_frequency, decay_rate)

    global_step = tf.train.get_or_create_global_step()

    train_step = tf.contrib.layers.optimize_loss(
        loss=total_loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer=optimizer,
        clip_gradients=clip_gradients,
        learning_rate_decay_fn=_lr_decay_fn)

    for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

    return train_step, global_step


def eval_accuracy_op(logits, batch_labels, k=5, prefix='category'):
    category_softmax = tf.nn.softmax(logits)
    predictions = tf.argmax(category_softmax, 1)
    labels = tf.argmax(batch_labels, 1)
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        prefix + '_Accuracy': tf.metrics.accuracy(predictions=predictions, labels=labels),
        prefix + '_Recall_%d' % k: tf.metrics.recall_at_k(predictions=category_softmax, labels=labels, k=k),
    })
    for name, value in names_to_values.items():
        summary_name = 'eval/%s' % name
        op = tf.summary.scalar(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    return names_to_updates

