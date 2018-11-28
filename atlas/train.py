import tensorflow as tf
from tensorflow import keras

tf.logging.set_verbosity(20)

def dataset_input_fn(tfrecord_path, batch_size, epochs=None, shuffle_buffer=100):

    def _parse_func(example_proto):
        # def _parse_image(image):
        #     image_decoded = tf.image.decode_png(image, channels=1)
        #     # image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        #     image_decoded = tf.squeeze(image_decoded, axis=2)
        #     return image_decoded

        features = {
            "image/rgby": tf.FixedLenFeature(shape=[], dtype=tf.string),
            "image/labels": tf.FixedLenFeature(shape=[28], dtype=tf.int64),
        }
        parsed_feature = tf.parse_single_example(example_proto, features)
        image = parsed_feature["image/rgby"]
        labels = parsed_feature["image/labels"]
        # image = tf.map_fn(_parse_image, image, dtype=tf.float32)
        # image_r = image[0] / 2 + image[3] / 2
        # image_g = image[1] / 2 + image[3] / 2
        # image = tf.stack([image_r, image_g, image[2]], 2)
        image = tf.decode_raw(image, out_type=tf.float32)
        image = tf.reshape(image, [299, 299, 3])
        # image = keras.applications.inception_resnet_v2.preprocess_input(tf.expand_dims(image, 0))
        # image = tf.image.resize_bicubic(tf.expand_dims(image, 0), size=[299, 299])
        # image = tf.squeeze(image, 0)
        # image = tf.transpose(image, [1, 2, 0])

        return image, labels

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=_parse_func, batch_size=batch_size))
    dataset = dataset.prefetch(32)
    if shuffle_buffer > 0:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buffer, count=epochs))
    else:
        dataset = dataset.repeat(epochs)
    return dataset


def f1(y_true, y_pred):
    tp = keras.backend.sum(y_true*y_pred, axis=0)
    fp = keras.backend.sum((1 - y_true)*y_pred, axis=0)
    fn = keras.backend.sum((1 - y_pred)*y_true, axis=0)

    p = tp / (tp + fp + keras.backend.epsilon())
    r = tp / (tp + fn + keras.backend.epsilon())

    f1 = 2*p*r / (p+r+keras.backend.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return keras.backend.mean(f1)


def loss_fn(y_true, y_pred):

    sigmoid_p = y_pred
    target_tensor = y_true

    zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = tf.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = tf.where(target_tensor > zeros, zeros, sigmoid_p)
    # per_entry_cross_ent = - 0.25 * (pos_p_sub ** 2) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
    #                       - (1 - 0.25) * (neg_p_sub ** 2) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    per_entry_cross_ent = - 0.25 * (pos_p_sub ** 2) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - 0.25 * (neg_p_sub ** 2) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def create_model():
    inp = keras.Input([299, 299, 3])
    base_model = keras.applications.InceptionResNetV2(include_top=False, input_tensor=inp)
    # for layer in base_model.layers:
    #     layer.trainable = False

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(28)(x)
    x = keras.layers.Activation('sigmoid')(x)

    model = keras.Model(inp, x)


    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.binary_crossentropy,
                  metrics=['acc', f1])
    # model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=loss_fn,
    #               metrics=['acc', f1])

    return model

def train_by_keras():
    train_dataset = dataset_input_fn("dataset/train.tfrecord", 16)
    val_dataset = dataset_input_fn("dataset/val.tfrecord", 16, shuffle_buffer=0)
    model = create_model()
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.00001, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint('/home/andy/Data/keras/atlas/keras.checkpoint', save_best_only=True),
        # tf.keras.callbacks.TensorBoard(log_dir='/home/andy/Data/keras/atlas/logs')
    ]
    model.fit(train_dataset, epochs=180, steps_per_epoch=1000, callbacks=callbacks, validation_data=val_dataset,
              validation_steps=20, verbose=1)


def train_by_estimator():
    model = create_model()
    config = tf.estimator.RunConfig(save_summary_steps=100, save_checkpoints_steps=100)
    model_estimator = tf.keras.estimator.model_to_estimator(keras_model=model, config=config, model_dir=model_dir)
    model_estimator.train(input_fn=lambda: dataset_input_fn("dataset/train.tfrecord", 16), max_steps=10000)


if __name__ == "__main__":
    model_dir = "/home/andy/Data/keras/atlas"
    # train_by_estimator()
    train_by_keras()
    

