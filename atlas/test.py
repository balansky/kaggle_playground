import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

def test_stack():
    img_b = Image.open("dataset/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png")
    img_g = Image.open("dataset/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png")
    img_r = Image.open("dataset/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png")
    img_y = Image.open("dataset/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png")
    imgs = [np.array(img_r), np.array(img_g), np.array(img_b), np.array(img_y)]
    imgs = np.stack(imgs, 2)
    test_im = Image.fromarray(imgs)
    test_im.save("dataset/tt.png")
    print('done')

def test_dataset():

    def _parse_func(example_proto):
        # def _parse_image(image):
        #     # image_decoded = tf.image.decode_png(image, channels=1)
        #     # image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        #     # image_decoded = tf.squeeze(image_decoded, axis=2)
        #     return image_decoded

        features = {
            "image/rgby": tf.FixedLenFeature(shape=[4], dtype=tf.string),
            "image/labels": tf.FixedLenFeature(shape=[28], dtype=tf.int64),
        }
        parsed_feature = tf.parse_single_example(example_proto, features)
        image = parsed_feature["image/rgby"]
        labels = parsed_feature["image/labels"]
        image = tf.decode_raw(image, out_type=tf.float32)
        # image = tf.map_fn(_parse_image, image, dtype=tf.float32)
        # image_r = image[0] / 2 + image[3] / 2
        # image_g = image[1] / 2 + image[3] / 2
        # image = tf.stack([image_r, image_g, image[2]], 2)
        # image = tf.keras.applications.inception_resnet_v2.preprocess_input(tf.expand_dims(image, 0))
        # image = tf.image.resize_bicubic(tf.expand_dims(image, 0), size=[299, 299])
        # image = tf.squeeze(image, 0)
        # image = tf.transpose(image, [1, 2, 0])

        return image, labels

    dataset = tf.data.TFRecordDataset("dataset/train.tfrecord")
    dataset = dataset.map(_parse_func)
    dataset = dataset.batch(5)
    iterator = dataset.make_initializable_iterator()
    next_images, next_labels = iterator.get_next()
    sess = tf.Session()
    sess.run(iterator.initializer)
    images, labels = sess.run([next_images, next_labels])
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    for i in range(5):
        ax[i].imshow(images[i])
    plt.show()
    print("done")

