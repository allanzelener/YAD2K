# YAD2K: Yet Another Darknet 2 Keras

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Welcome to YAD2K

You only look once, but you reimplement neural nets over and over again.

YAD2K is a 90% Keras/10% Tensorflow implementation of YOLO_v2.

Original paper: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by Joseph Redmond and Ali Farhadi.

![YOLO_v2 COCO model with test_yolo defaults](images/out/dog_small.jpg)

--------------------------------------------------------------------------------

## Requirements

- [Keras](https://github.com/fchollet/keras)
- [Tensorflow](https://www.tensorflow.org/) r0.12
- [Numpy](http://www.numpy.org/)
- [Pillow](https://pillow.readthedocs.io/) (for rendering test results)
- [Python](https://www.python.org/) 3.5

With conda use `conda env create -f environment.yml` to replicate the development environment. This environment is likely to be overcomplete.

## Quick Start

- Download Darknet model weights from the [official YOLO website](http://pjreddie.com/darknet/yolo/).
- Convert the Darknet YOLO_v2 model to a Keras model.
- Test the converted model on the small test set in `images/`.

```bash
wget http://pjreddie.com/media/files/yolo.weights
./yad2k.py cfg/yolo.cfg yolo.weights model_data/yolo.h5
./test_yolo.py model_data/yolo.h5  # output in images/out/
```

See `./yad2k.py --help` and `./test_yolo.py --help` for more options.

--------------------------------------------------------------------------------

## More Details

The YAD2K converter currently only supports YOLO_v2 style models, this include the following configurations: `darknet19_448`, `tiny-yolo-voc`, `yolo-voc`, and `yolo`.

`yad2k.py -p` will produce a plot of the generated Keras model. For example see [yolo.png](model_data/yolo.png).

YAD2K assumes the Keras backend is Tensorflow. In particular for YOLO_v2 models with a passthrough layer, YAD2K uses `tf.space_to_depth` to implement the passthrough layer. The evaluation script also directly uses Tensorflow tensors and uses `tf.non_max_suppression` for the final output.

`voc_conversion_scripts` contains two scripts for converting the Pascal VOC image dataset with XML annotations to either HDF5 or TFRecords format for easier training with Keras or Tensorflow.

`yad2k/models` contains reference implementations of Darknet-19 and YOLO_v2.

## Known Issues and TODOs

- Error deserializing Lambda wrapping space_to_depth. Apply [this PR to Keras](https://github.com/fchollet/keras/pull/5350).
- Add YOLO_v2 loss function.
- Script to train YOLO_v2 reference model.
- Support for additional Darknet layer types.
- Tuck away the Tensorflow dependencies with Keras wrappers where possible.

## Darknets of Yore

YAD2K stands on the shoulders of giants.

- :fire: [Darknet](https://github.com/pjreddie/darknet) :fire:
- [Darknet.Keras](https://github.com/sunshineatnoon/Darknet.keras) - The original D2K for YOLO_v1.
- [Darkflow](https://github.com/thtrieu/darkflow) - Darknet directly to Tensorflow.
- [caffe-yolo](https://github.com/xingwangsfu/caffe-yolo) - YOLO_v1 to Caffe.

--------------------------------------------------------------------------------
