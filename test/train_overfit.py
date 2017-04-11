import io
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model, load_model

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

COCO_ANCHORS = np.array(
    ((0.738768, 0.874946), (2.42204, 2.65704), (4.30971, 7.04493),
     (10.246, 4.59428), (12.6868, 11.8741)))


def main():
    voc_path = os.path.expanduser('~/datasets/VOCdevkit/pascal_voc_07_12.hdf5')
    classes_path = os.path.expanduser('model_data/pascal_classes.txt')
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    voc = h5py.File(voc_path, 'r')
    image = PIL.Image.open(io.BytesIO(voc['train/images'][28]))
    orig_size = np.array([image.width, image.height])
    orig_size = np.expand_dims(orig_size, axis=0)
    image = image.resize((416, 416), PIL.Image.BICUBIC)
    image_data = np.array(image, dtype=np.float)
    image_data /= 255.
    boxes = voc['train/boxes'][28]
    boxes = boxes.reshape((-1, 5))
    boxes_extents = boxes[:, [2, 1, 4, 3, 0]]
    boxes_xy = 0.5 * (boxes[:, 3:5] + boxes[:, 1:3])
    boxes_wh = boxes[:, 3:5] - boxes[:, 1:3]
    boxes_xy = boxes_xy / orig_size
    boxes_wh = boxes_wh / orig_size
    boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 0:1]), axis=1)
    image_data = np.expand_dims(image_data, axis=0)
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    box_flags_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)
    box_flags_input = Input(shape=box_flags_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)
    anchors = COCO_ANCHORS
    print(anchors.shape, boxes.shape)

    box_flags, matching_true_boxes = preprocess_true_boxes(boxes, anchors,
                                                           [416, 416])
    print(boxes)
    print(boxes_extents)
    print(np.where(box_flags == 1)[:-1])
    print(matching_true_boxes[np.where(box_flags == 1)[:-1]])
    # from IPython import embed
    # embed()

    model_body = yolo_body(image_input, len(anchors), len(class_names))
    model_body = Model(image_input, model_body.output)
    with tf.device('/cpu:0'):
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input, box_flags_input,
                           matching_boxes_input
                       ])
    model = Model(
        [image_input, boxes_input, box_flags_input,
         matching_boxes_input], model_loss)
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function.

    boxes = np.expand_dims(boxes, axis=0)
    box_flags = np.expand_dims(box_flags, axis=0)
    matching_true_boxes = np.expand_dims(matching_true_boxes, axis=0)

    num_steps = 10000
    # for i in range(num_steps):
    #     loss = model.train_on_batch(
    #         [image_data, boxes, box_flags, matching_true_boxes], np.zeros(len(image_data)))
    #     print(i, loss)
    model.fit([image_data, boxes, box_flags, matching_true_boxes],
              np.zeros(len(image_data)),
              batch_size=1,
              nb_epoch=num_steps)
    model.save_weights('overfit_weights.h5')

    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=.3, iou_threshold=.9)

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            model_body.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })
    print('Found {} boxes for image.'.format(len(out_boxes)))
    print(out_boxes)
    image_with_boxes = draw_boxes(image_data[0], out_boxes, out_classes,
                                  class_names, out_scores)
    plt.imshow(image_with_boxes, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    main()
