"""Convert Pascal VOC 2007+2012 detection dataset to TFRecords.
Does not preserve full XML annotations.
Combines all VOC 2007 subsets (train, val) with VOC2012 for training.
Uses VOC2012 val for val and VOC2007 test for test.

Code based on:
https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
"""

import argparse
import os
import xml.etree.ElementTree as ElementTree
from datetime import datetime

import numpy as np
import tensorflow as tf

from voc_to_hdf5 import get_ids

sets_from_2007 = [('2007', 'train'), ('2007', 'val')]
train_set = [('2012', 'train'), ('2012', 'val')]
test_set = [('2007', 'test')]

classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

parser = argparse.ArgumentParser(
    description='Convert Pascal VOC 2007+2012 detection dataset to TFRecords.')
parser.add_argument(
    '-p',
    '--path_to_voc',
    help='path to Pascal VOC dataset',
    default='~/data/PascalVOC/VOCdevkit')

# Small graph for image decoding
decoder_sess = tf.Session()
image_placeholder = tf.placeholder(dtype=tf.string)
decoded_jpeg = tf.image.decode_jpeg(image_placeholder, channels=3)


def process_image(image_path):
    """Decode image at given path."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    image = decoder_sess.run(decoded_jpeg,
                             feed_dict={image_placeholder: image_data})
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[2]
    assert image.shape[2] == 3
    return image_data, height, width


def process_anno(anno_path):
    """Process Pascal VOC annotations."""
    with open(anno_path) as f:
        xml_tree = ElementTree.parse(f)
    root = xml_tree.getroot()
    size = root.find('size')
    height = float(size.find('height').text)
    width = float(size.find('width').text)
    boxes = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        label = obj.find('name').text
        if label not in classes or int(
                difficult) == 1:  # exclude difficult or unlisted classes
            continue
        xml_box = obj.find('bndbox')
        bbox = {
            'class': classes.index(label),
            'y_min': float(xml_box.find('ymin').text) / height,
            'x_min': float(xml_box.find('xmin').text) / width,
            'y_max': float(xml_box.find('ymax').text) / height,
            'x_max': float(xml_box.find('xmax').text) / width
        }
        boxes.append(bbox)
    return boxes


def convert_to_example(image_data, boxes, filename, height, width):
    """Convert Pascal VOC ground truth to TFExample protobuf.

    Parameters
    ----------
    image_data : bytes
        Encoded image bytes.
    boxes : dict
        Bounding box corners and class labels
    filename : string
        Path to image file.
    height : int
        Image height.
    width : int
        Image width.

    Returns
    -------
    example : protobuf
        Tensorflow Example protobuf containing image and bounding boxes.
    """
    box_classes = [b['class'] for b in boxes]
    box_ymin = [b['y_min'] for b in boxes]
    box_xmin = [b['x_min'] for b in boxes]
    box_ymax = [b['y_max'] for b in boxes]
    box_xmax = [b['x_max'] for b in boxes]
    encoded_image = [tf.compat.as_bytes(image_data)]
    base_name = [tf.compat.as_bytes(os.path.basename(filename))]

    example = tf.train.Example(features=tf.train.Features(feature={
        'filename':
        tf.train.Feature(bytes_list=tf.train.BytesList(value=base_name)),
        'height':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'classes':
        tf.train.Feature(int64_list=tf.train.Int64List(value=box_classes)),
        'y_mins':
        tf.train.Feature(float_list=tf.train.FloatList(value=box_ymin)),
        'x_mins':
        tf.train.Feature(float_list=tf.train.FloatList(value=box_xmin)),
        'y_maxes':
        tf.train.Feature(float_list=tf.train.FloatList(value=box_ymax)),
        'x_maxes':
        tf.train.Feature(float_list=tf.train.FloatList(value=box_xmax)),
        'encoded':
        tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_image))
    }))
    return example


def get_image_path(voc_path, year, image_id):
    """Get path to image for given year and image id."""
    return os.path.join(voc_path, 'VOC{}/JPEGImages/{}.jpg'.format(year,
                                                                   image_id))


def get_anno_path(voc_path, year, image_id):
    """Get path to image annotation for given year and image id."""
    return os.path.join(voc_path, 'VOC{}/Annotations/{}.xml'.format(year,
                                                                    image_id))


def process_dataset(name, image_paths, anno_paths, result_path, num_shards):
    """Process selected Pascal VOC dataset to generate TFRecords files.

    Parameters
    ----------
    name : string
        Name of resulting dataset 'train' or 'test'.
    image_paths : list
        List of paths to images to include in dataset.
    anno_paths : list
        List of paths to corresponding image annotations.
    result_path : string
        Path to put resulting TFRecord files.
    num_shards : int
        Number of shards to split TFRecord files into.
    """
    shard_ranges = np.linspace(0, len(image_paths), num_shards + 1).astype(int)
    counter = 0
    for shard in range(num_shards):
        # Generate shard file name
        output_filename = '{}-{:05d}-of-{:05d}'.format(name, shard, num_shards)
        output_file = os.path.join(result_path, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = range(shard_ranges[shard], shard_ranges[shard + 1])
        for i in files_in_shard:
            image_file = image_paths[i]
            anno_file = anno_paths[i]

            # processes image + anno
            image_data, height, width = process_image(image_file)
            boxes = process_anno(anno_file)

            # convert to example
            example = convert_to_example(image_data, boxes, image_file, height,
                                         width)

            # write to writer
            writer.write(example.SerializeToString())

            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('{} : Processed {:d} of {:d} images.'.format(
                    datetime.now(), counter, len(image_paths)))
        writer.close()
        print('{} : Wrote {} images to {}'.format(
            datetime.now(), shard_counter, output_filename))

    print('{} : Wrote {} images to {} shards'.format(datetime.now(), counter,
                                                     num_shards))


def _main(args):
    """Locate files for train and test sets and then generate TFRecords."""
    voc_path = args.path_to_voc
    voc_path = os.path.expanduser(voc_path)
    result_path = os.path.join(voc_path, 'TFRecords')
    print('Saving results to {}'.format(result_path))

    train_path = os.path.join(result_path, 'train')
    test_path = os.path.join(result_path, 'test')

    train_ids = get_ids(voc_path, train_set)  # 2012 trainval
    test_ids = get_ids(voc_path, test_set)  # 2007 test
    train_ids_2007 = get_ids(voc_path, sets_from_2007)  # 2007 trainval
    total_train_ids = len(train_ids) + len(train_ids_2007)
    print('{} train examples and {} test examples'.format(total_train_ids,
                                                          len(test_ids)))

    train_image_paths = [
        get_image_path(voc_path, '2012', i) for i in train_ids
    ]
    train_image_paths.extend(
        [get_image_path(voc_path, '2007', i) for i in train_ids_2007])
    test_image_paths = [get_image_path(voc_path, '2007', i) for i in test_ids]

    train_anno_paths = [get_anno_path(voc_path, '2012', i) for i in train_ids]
    train_anno_paths.extend(
        [get_anno_path(voc_path, '2007', i) for i in train_ids_2007])
    test_anno_paths = [get_anno_path(voc_path, '2007', i) for i in test_ids]

    process_dataset(
        'train',
        train_image_paths,
        train_anno_paths,
        train_path,
        num_shards=60)
    process_dataset(
        'test', test_image_paths, test_anno_paths, test_path, num_shards=20)


if __name__ == '__main__':
    _main(parser.parse_args(args))
