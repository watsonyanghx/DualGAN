# DualGAN in TensorFlow

This code is modified from [duxingren14/DualGAN](https://github.com/duxingren14/DualGAN), in order to make it less coupled.

There are 2 changes:

1. Split up training process from DualGAN model part. 

    In origin code, the training operation is excuted inside DualGAN model part, this make it hard to change runing mode of Batch Normalization (Batch Normalization has different behavior in training and inference phase).

2. Set `is_training=True` in **training phase** and set `is_training=False` in **inference phase**. Have a look at [ops.py](https://github.com/watsonyanghx/DualGAN_Tensorflow/blob/master/ops.py#L26).

    In origin code, the Batch Normalization is always running in **training mode** even when sampling images, which is supposed to be set to `is_training=False` as described above.


## 

I did't mean to generate very realistic images that can fool human, I just want to make it clear **how it is different** of generated images when using [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf), [Batch Renormalization](https://arxiv.org/pdf/1702.03275v2.pdf), [Instance Normalization](https://arxiv.org/pdf/1607.08022v2.pdf).

As a result, the generated images looks bad, but after carefully tuning model and training parameters, good results can also be generated.


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

    All of images in [duxingren14/DualGAN](https://github.com/duxingren14/DualGAN) are generated in **training mode**, which means setting `is_training=True` when sampling images.


2. Inference mode

    However, in inference phase, the generated images looks very bad, there are just noisy points. 

    Images generated in inference mode:

    ![]()


    I don't know what exactly is going on here. Someone could explain this ?


## Update

1. It seems that the bad results generated in inference mode is caused by Batch Normalization, see links bellow for explaination.

    - [What if Batch Normalization is used in training mode when testing?](https://stackoverflow.com/questions/46290930/what-if-batch-normalization-is-used-in-training-mode-when-testing)

2. To solve this problem, you can use [Batch Renormalization](https://arxiv.org/pdf/1702.03275v2.pdf) or [Instance Normalization](https://arxiv.org/pdf/1607.08022v2.pdf).

3. I tried Batch Renormalization ([tf.layers.batch_normalization](https://www.tensorflow.org/versions/master/api_docs/python/tf/layers/batch_normalization) in  tensorflow), and the results on dataset are much better than that generated using Batch Normalization, but results on other datasets are as bad as Batch Normalization. Maybe the hyper-parameters (`renorm_clipping`, `renorm_momentum` and other model or training parameters) are what matters.

4. Instance Normalization is not test, Instance Normalization is supposed to get as good as runing in training mode in Batch Normalization. You can have a try and see what will happen.
