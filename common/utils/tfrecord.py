import tensorflow as tf


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def sequence_feature(dict_data):
    return tf.train.FeatureLists(feature_list={
        k: tf.train.FeatureList(feature=v) for k, v in dict_data.items()
    })


def train_feature(features):
    return tf.train.Features(feature=features)


def sequence_example(context_features, sequence_features):
    return tf.train.SequenceExample(context=context_features,
                                    feature_lists=sequence_features)


def train_example(features):
    return tf.train.Example(features=tf.train.Features(feature=features))


def get_writer(output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    return writer