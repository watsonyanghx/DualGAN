# DualGAN in TensorFlow

This code is modifreied from [duxingren14/DualGAN](https://github.com/duxingren14/DualGAN), in order to make it less coupled.

There are 2 changes:

1. Split up training process from DualGAN model part. 

  In origin code, the training operation is excuted inside DualGAN model part, this make it hard to change runing mode of Batch Normalization (Batch Normalization has different behavior in training and testing phase).

2. Set `is_training=True` in training phase and set `is_training=False` in infering phase. 

  In origin code, the batch normalization is always running in training mode even when sampling images, which is supposed to set `is_training=False` as described above.


## Data

The data could be downloaded from [duxingren14/DualGAN](https://github.com/duxingren14/DualGAN).


## How to run


THe parameters could be changed from []().

1. Train

`python main.py --phase train --dataset_name day-night`


2. Inference

`python main.py --phase infer --dataset_name day-night`


## Result

1. Train

The generated images in training mode still look good. More samples could be seen at [duxingren14/DualGAN](https://github.com/duxingren14/DualGAN). 

Images generated in train mode:

![]()


**Note:**

All of images in [duxingren14/DualGAN](https://github.com/duxingren14/DualGAN) are generated in training mode, meaning set `is_training=True` when sampling images.


2. Inference

However, in inference phase, the generated images looks very bad, there are just noisy points. 

Images generated in inference mode:

![]()


I don't know what exactly is going on here. Someone could explain this ?

