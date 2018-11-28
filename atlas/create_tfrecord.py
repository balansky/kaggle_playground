import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import pandas
import logging
import os

logging.basicConfig(level=20)


def load_data(dataset_dir):
    label_cv = pandas.read_csv(os.path.join(dataset_dir, 'train.csv'))
    data = []
    for name, targets in zip(label_cv['Id'], label_cv['Target']):
        img_stacks = []
        try:
            for color_postfix in ['_red.png', '_green.png', '_blue.png', '_yellow.png']:
                img_path = os.path.join(dataset_dir, 'train', name + color_postfix)
                if os.path.exists(img_path):
                    img_stacks.append(img_path)
                else:
                    raise FileNotFoundError(img_path + ": Not Found!")
        except Exception as err:
            logging.warning(str(err))
            continue
        labels = [int(t) for t in targets.split(' ')]
        data.append((img_stacks, labels, name))
    return data


def split_train_val(data_set, validate_fraction):
    shuffled_index = list(range(len(data_set)))
    np.random.shuffle(shuffled_index)
    shuffled_dataset = [data_set[i] for i in shuffled_index]
    val_num = int(len(data_set)*validate_fraction)
    train_num = len(data_set) - val_num
    return shuffled_dataset[:train_num], shuffled_dataset[train_num:]


def preprocess_image(stacked_images):
    imgs = []
    for image_path in stacked_images:
        imgs.append(np.asarray(Image.open(image_path).resize((299, 299)), dtype=np.float32))
    image_r = imgs[0] / 2 + imgs[3] / 2
    image_g = imgs[1] / 2 + imgs[3] / 2
    image_b = imgs[2]
    img = np.stack([image_r, image_g, image_b], axis=2)
    img /= 127.5
    img -= 1.
    return img


def create_tfrecord(data, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    logging.info("Creating " + output_path)
    success = 0
    for i, (images, labels, name) in enumerate(data):
        try:
            l = np.zeros(28, dtype=np.int)
            l[labels] = 1
            img = preprocess_image(images)
            features = {
                "image/rgby": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                "image/labels": tf.train.Feature(int64_list=tf.train.Int64List(value=l))
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
            success += 1
            # if i >= 9:
            #     break
        except Exception as err:
            logging.warning(str(err))
    writer.close()
    logging.info("Total Conversion: %d" % success)


def main(dataset_dir, validate_fraction):
    data = load_data(dataset_dir)
    train_data, val_data = split_train_val(data, validate_fraction)
    create_tfrecord(data, os.path.join(dataset_dir, "train.tfrecord"))
    create_tfrecord(val_data, os.path.join(dataset_dir, "val.tfrecord"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default='dataset',
        type=str
    )
    parser.add_argument(
        "--validate_fraction",
        type=float,
        default=0.2
    )
    args, unparsed = parser.parse_known_args()
    main(args.dataset_dir, args.validate_fraction)