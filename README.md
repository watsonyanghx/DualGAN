# DualGAN in TensorFlow

This code is modified from [duxingren14/DualGAN](https://github.com/duxingren14/DualGAN), in order to make it less coupled.

There are 2 changes:

1. Split up training process from DualGAN model part. 

    In origin code, the training operation is excuted inside DualGAN model part, this make it hard to change runing mode of Batch Normalization (Batch Normalization has different behavior in training and testing phase).

2. Set `is_training=True` in training phase and set `is_training=False` in infering phase. Have a look at [ops.py](https://github.com/watsonyanghx/DualGAN_Tensorflow/blob/master/ops.py#L26).

    In origin code, the Batch Normalization is always running in training mode even when sampling images, which is supposed to be set to `is_training=False` as described above.


## Prerequisite

- Numpy

- TensorFlow 1.2 or higher

- scipy.misc


## Datasets

Datasets could be downloaded from [duxingren14/DualGAN](https://github.com/duxingren14/DualGAN).


## How to run

There are many parameters with which you can play, have a look at [main.py](https://github.com/watsonyanghx/DualGAN_Tensorflow/blob/master/main.py#L21).

1. Train

    `python main.py --phase train --dataset_name day-night`


2. Inference

    `python main.py --phase infer --dataset_name day-night`


## Result

1. Training mode

    The generated images in training mode still look good. More samples could be seen at [duxingren14/DualGAN](https://github.com/duxingren14/DualGAN). 

    Images generated in train mode:

    ![]()


    **Note:**

    All of images in [duxingren14/DualGAN](https://github.com/duxingren14/DualGAN) are generated in **training mode**, meaning set `is_training=True` when sampling images.


2. Inference mode

    However, in inference phase, the generated images looks very bad, there are just noisy points. 

    Images generated in inference mode:

    ![]()


    I don't know what exactly is going on here. Someone could explain this ?

