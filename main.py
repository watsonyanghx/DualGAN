"""

"""

import numpy as np
import tensorflow as tf

import argparse
import os
import time
import six.moves
from glob import glob

from model import DualNet
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
# parser.add_argument('--network_type', dest='network_type', default='fcn_4',
#                     help='fcn_1,fcn_2,fcn_4,fcn_8, fcn_16, fcn_32, fcn_64, fcn_128')
parser.add_argument('--image_size', dest='image_size', type=int, default=128,
                    help='size of input images (applicable to both A images and B images)')
parser.add_argument('--fcn_filter_dim', dest='fcn_filter_dim', type=int, default=64,
                    help='# of fcn filters in first conv layer')
parser.add_argument('--input_channels_A', dest='input_channels_A', type=int, default=3,
                    help='# of input image channels')
parser.add_argument('--input_channels_B', dest='input_channels_B', type=int, default=3,
                    help='# of output image channels')

"""Arguments related to run mode"""
parser.add_argument('--phase', dest='phase', default='train', help='train, test')

"""Arguments related to training"""
parser.add_argument('--loss_metric', dest='loss_metric', default='L1', help='L1, or L2')
parser.add_argument('--niter', dest='niter', type=int, default=30,
                    help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.00005,
                    help='initial learning rate for adam')  # 0.0002
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5,
                    help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True,
                    help='if flip the images for data argumentation')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades',
                    help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1,
                    help='# images in batch')
parser.add_argument('--lambda_A', dest='lambda_A', type=float, default=200.0,
                    help='# weights of A recovery loss')
parser.add_argument('--lambda_B', dest='lambda_B', type=float, default=200.0,
                    help='# weights of B recovery loss')
parser.add_argument('--n_critic', dest='n_critic', type=int, default=3, help='#n_critic')
parser.add_argument('--clamp', dest='clamp', type=float, default=0.01, help='#n_critic')

"""Arguments related to monitoring and outputs"""
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50,
                    help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=1000,
                    help='save the latest model every latest_freq sgd iterations '
                         '(overwrites the previous latest model)')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint',
                    help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./train_sample',
                    help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test_sample',
                    help='test sample are saved here')

args_ = parser.parse_args()


def load(saver, sess, checkpoint_dir, model_dir):
    print(" [*] Reading checkpoint...")

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False


def save(saver, sess, checkpoint_dir, model_dir, step):
    model_name = "DualNet.model"
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess,
               os.path.join(checkpoint_dir, model_name),
               global_step=step)


def load_training_imgs(data, idx, batch_size, image_size, flip):
    batch_files = data[idx * batch_size:(idx + 1) * batch_size]
    batch = [utils.load_data(batch_file, image_size=image_size, flip=flip)
             for batch_file in batch_files]

    batch_images = np.reshape(np.array(batch).astype(np.float32),
                              (batch_size, image_size, image_size, -1))

    return batch_images


def load_random_samples(dataset_name, batch_size, image_size, mode):
    data = np.random.choice(glob('./datasets/{}/{}/A/*.jpg'.format(dataset_name, mode)),
                            batch_size)
    sample_A = [utils.load_data(sample_file, image_size=image_size, flip=False)
                for sample_file in data]

    data = np.random.choice(glob('./datasets/{}/{}/B/*.jpg'.format(dataset_name, mode)),
                            batch_size)
    sample_B = [utils.load_data(sample_file, image_size=image_size, flip=False)
                for sample_file in data]

    sample_A_images = np.reshape(np.array(sample_A).astype(np.float32),
                                 (batch_size, image_size, image_size, -1))
    sample_B_images = np.reshape(np.array(sample_B).astype(np.float32),
                                 (batch_size, image_size, image_size, -1))
    return sample_A_images, sample_B_images


def infer(args, dir_name, epoch):
    model = DualNet(is_training=False,
                    image_size=args.image_size,
                    batch_size=args.batch_size,
                    fcn_filter_dim=args.fcn_filter_dim,
                    input_channels_A=args.input_channels_A,
                    input_channels_B=args.input_channels_B,
                    dataset_name=args.dataset_name,
                    lambda_A=args.lambda_A,
                    lambda_B=args.lambda_B,
                    loss_metric=args.loss_metric)

    sample_A_imgs, sample_B_imgs = load_random_samples(args.dataset_name,
                                                       args.batch_size,
                                                       args.image_size,
                                                       'test')

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        if load(saver, sess, args.checkpoint_dir, dir_name):
            print(" [*] Load SUCCESS")
        else:
            print(" Load failed...neglected")
            print(" start training...")

        Ag, recover_A_value, translated_A_value = \
            sess.run([model.A_loss, model.recover_A, model.translated_A],
                     feed_dict={model.real_A: sample_A_imgs,
                                model.real_B: sample_B_imgs})

        Bg, recover_B_value, translated_B_value = \
            sess.run([model.B_loss, model.recover_B, model.translated_B],
                     feed_dict={model.real_A: sample_A_imgs,
                                model.real_B: sample_B_imgs})

        utils.save_images(translated_A_value, [args.batch_size, 1],
                          './{}/{}/{:06d}_test_translated_A.png'.
                          format(args.test_dir, dir_name, epoch))
        utils.save_images(recover_A_value, [args.batch_size, 1],
                          './{}/{}/{:06d}_test_recover_A_.png'.
                          format(args.test_dir, dir_name, epoch))

        utils.save_images(translated_B_value, [args.batch_size, 1],
                          './{}/{}/{:06d}_test_translated_B.png'.
                          format(args.test_dir, dir_name, epoch))
        utils.save_images(recover_B_value, [args.batch_size, 1],
                          './{}/{}/{:06d}_test_recover_B_.png'.
                          format(args.test_dir, dir_name, epoch))

        print("[Sample] A_loss: {:.8f}, B_loss: {:.8f}".format(Ag, Bg))


def train(args, dir_name):
    """Train Dual GAN"""
    model = DualNet(is_training=True,
                    image_size=args.image_size,
                    batch_size=args.batch_size,
                    fcn_filter_dim=args.fcn_filter_dim,
                    input_channels_A=args.input_channels_A,
                    input_channels_B=args.input_channels_B,
                    dataset_name=args.dataset_name,
                    lambda_A=args.lambda_A,
                    lambda_B=args.lambda_B,
                    loss_metric=args.loss_metric)

    decay = 0.9
    d_optim = tf.train.RMSPropOptimizer(args.lr, decay=decay). \
        minimize(model.d_loss, var_list=model.d_vars)

    g_optim = tf.train.RMSPropOptimizer(args.lr, decay=decay). \
        minimize(model.g_loss, var_list=model.g_vars)

    clip_d_vars_ops = [val.assign(tf.clip_by_value(val, -args.clamp, args.clamp))
                       for val in model.d_vars]

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        tf.summary.FileWriter("./logs/" + dir_name, sess.graph)

        counter = 1
        start_time = time.time()

        if load(saver, sess, args.checkpoint_dir, dir_name):
            print(" [*] Load SUCCESS")
        else:
            print(" Load failed...neglected")
            print(" start training...")

        for epoch in six.moves.xrange(args.epoch):
            data_A = glob('./datasets/{}/train/A/*.jpg'.format(args.dataset_name))
            data_B = glob('./datasets/{}/train/B/*.jpg'.format(args.dataset_name))
            np.random.shuffle(data_A)
            np.random.shuffle(data_B)
            batch_idxs = min(len(data_A), len(data_B)) // (args.batch_size * args.n_critic)
            print('[*] training data loaded successfully')
            print("#data_A: %d  #data_B:%d" % (len(data_A), len(data_B)))
            print('[*] run optimizor...')

            for idx in six.moves.range(0, batch_idxs):
                imgA_batch_list = \
                    [load_training_imgs(data_A, idx + i, args.batch_size, args.image_size, args.flip)
                     for i in six.moves.range(args.n_critic)]
                imgB_batch_list = \
                    [load_training_imgs(data_B, idx + i, args.batch_size, args.image_size, args.flip)
                     for i in six.moves.range(args.n_critic)]

                print("Epoch: [%2d] [%4d/%4d] [%2d]" % (epoch, idx, batch_idxs, counter))
                counter = counter + 1

                # run_optim(args, imgA_batch_list, imgB_batch_list, counter, start_time)
                for i in six.moves.range(args.n_critic):
                    batch_A_images = imgA_batch_list[i]
                    batch_B_images = imgB_batch_list[i]
                    _, Adfake, Adreal, Bdfake, Bdreal, Ad, Bd, summary_str = \
                        sess.run([d_optim, model.A_d_loss_fake, model.A_d_loss_real,
                                  model.B_d_loss_fake, model.B_d_loss_real, model.A_d_loss,
                                  model.B_d_loss, model.d_loss_sum],
                                 feed_dict={model.real_A: batch_A_images,
                                            model.real_B: batch_B_images})
                    # self.writer.add_summary(summary_str, counter)
                    sess.run(clip_d_vars_ops)

                batch_A_images = imgA_batch_list[np.random.randint(args.n_critic, size=1)[0]]
                batch_B_images = imgB_batch_list[np.random.randint(args.n_critic, size=1)[0]]
                _, Ag, Bg, Aloss, Bloss, summary_str = \
                    sess.run([g_optim, model.A_g_loss, model.B_g_loss, model.A_loss,
                              model.B_loss, model.g_loss_sum],
                             feed_dict={model.real_A: batch_A_images,
                                        model.real_B: batch_B_images})

                # self.writer.add_summary(summary_str, counter)
                print("time: %4.4f, A_d_loss: %.2f, A_g_loss: %.2f, B_d_loss: %.2f, "
                      "B_g_loss: %.2f,  A_loss: %.5f, B_loss: %.5f"
                      % (time.time() - start_time, Ad, Ag, Bd, Bg, Aloss, Bloss))
                print("A_d_loss_fake: %.2f, A_d_loss_real: %.2f, B_d_loss_fake: %.2f, "
                      "B_g_loss_real: %.2f"
                      % (Adfake, Adreal, Bdfake, Bdreal))
                # end of run_optim

                if np.mod(counter, 100) == 1:
                    # sample_shotcut(args.sample_dir, epoch, idx, batch_idxs)
                    sample_A_imgs, sample_B_imgs = load_random_samples(args.dataset_name,
                                                                       args.batch_size,
                                                                       args.image_size,
                                                                       'val')

                    Ag, recover_A_value, translated_A_value = \
                        sess.run([model.A_loss, model.recover_A, model.translated_A],
                                 feed_dict={model.real_A: sample_A_imgs,
                                            model.real_B: sample_B_imgs})

                    Bg, recover_B_value, translated_B_value = \
                        sess.run([model.B_loss, model.recover_B, model.translated_B],
                                 feed_dict={model.real_A: sample_A_imgs,
                                            model.real_B: sample_B_imgs})

                    utils.save_images(translated_A_value, [args.batch_size, 1],
                                      './{}/{}/{:06d}_{:04d}_train_translated_A_{:02d}.png'.
                                      format(args.sample_dir, dir_name, epoch, idx, batch_idxs))
                    utils.save_images(recover_A_value, [args.batch_size, 1],
                                      './{}/{}/{:06d}_{:04d}_train_recover_A_{:02d}_.png'.
                                      format(args.sample_dir, dir_name, epoch, idx, batch_idxs))

                    utils.save_images(translated_B_value, [args.batch_size, 1],
                                      './{}/{}/{:06d}_{:04d}_train_translated_B_{:02d}.png'.
                                      format(args.sample_dir, dir_name, epoch, idx, batch_idxs))
                    utils.save_images(recover_B_value, [args.batch_size, 1],
                                      './{}/{}/{:06d}_{:04d}_train_recover_B_epoch={:02d}.png'.
                                      format(args.sample_dir, dir_name, epoch, idx, batch_idxs))

                    print("[Sample] A_loss: {:.8f}, B_loss: {:.8f}".format(Ag, Bg))
                    # end of sample_shotcut

                if np.mod(counter, args.save_latest_freq) == 2:
                    save(saver, sess, args.checkpoint_dir, dir_name, counter)


def main(_):
    if not os.path.exists(args_.checkpoint_dir):
        os.makedirs(args_.checkpoint_dir)
    if not os.path.exists(args_.sample_dir):
        os.makedirs(args_.sample_dir)
    if not os.path.exists(args_.test_dir):
        os.makedirs(args_.test_dir)

    # directory name for output and logs saving
    dir_name = \
        "%s-batch_sz_%s-img_sz_%s-fltr_dim_%d-%s-lambda_AB_%s_%s-c_%s-n_critic_%s" % \
        (args_.dataset_name, args_.batch_size, args_.image_size, args_.fcn_filter_dim,
         args_.loss_metric, args_.lambda_A, args_.lambda_B, args_.clamp, args_.n_critic)

    if args_.phase == 'train':
        train(args_, dir_name)
    else:
        epoch = 50
        infer(args_, dir_name, epoch)


if __name__ == '__main__':
    tf.app.run()
