"""

"""

from ops import *


class DualNet(object):
    def __init__(self,
                 is_training=True,
                 image_size=128,
                 batch_size=1,
                 fcn_filter_dim=64,
                 input_channels_A=3,
                 input_channels_B=3,
                 dataset_name='facades',
                 lambda_A=200,
                 lambda_B=200,
                 loss_metric='L1'):
        """
        Args:
            is_training:
            batch_size: The size of batch. Should be specified before training. [1]
            image_size: (optional) The resolution in pixels of the images. [128]
            fcn_filter_dim: (optional) Dimension of fcn filters in first conv layer. [64]
            input_channels_A: (optional) Dimension of input image color of Network A.
                For grayscale input, set to 1. [3]
            input_channels_B: (optional) Dimension of output image color of Network B.
                For grayscale input, set to 1. [3]
        """
        self.is_training = is_training

        self.df_dim = fcn_filter_dim

        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

        self.batch_size = batch_size
        self.image_size = image_size

        self.fcn_filter_dim = fcn_filter_dim

        self.input_channels_A = input_channels_A
        self.input_channels_B = input_channels_B
        self.loss_metric = loss_metric

        self.dataset_name = dataset_name

        self.build_model()

    def build_model(self):
        # define place holders
        self.real_A = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_size, self.image_size, self.input_channels_A],
                                     name='input_images_of_A_network')
        self.real_B = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_size, self.image_size, self.input_channels_B],
                                     name='input_images_of_B_network')

        # define graphs
        self.translated_A = self.A_g_net(self.real_A, reuse=False)
        self.A_D_predictions = self.A_d_net(self.translated_A, reuse=False)

        self.translated_B = self.B_g_net(self.real_B, reuse=False)
        self.B_D_predictions = self.B_d_net(self.translated_B, reuse=False)

        # define loss
        self.recover_A = self.B_g_net(self.translated_A, reuse=True)
        self.recover_B = self.A_g_net(self.translated_B, reuse=True)

        if self.loss_metric == 'L1':
            self.A_loss = tf.reduce_mean(tf.abs(self.recover_A - self.real_A))
            self.B_loss = tf.reduce_mean(tf.abs(self.recover_B - self.real_B))
        elif self.loss_metric == 'L2':
            self.A_loss = tf.reduce_mean(tf.square(self.recover_A - self.real_A))
            self.B_loss = tf.reduce_mean(tf.square(self.recover_B - self.real_B))

        self.A_D_predictions_ = self.A_d_net(self.real_B, reuse=True)
        self.A_d_loss_real = tf.reduce_mean(-self.A_D_predictions_)
        self.A_d_loss_fake = tf.reduce_mean(self.A_D_predictions)

        self.A_d_loss = self.A_d_loss_fake + self.A_d_loss_real
        self.A_g_loss = tf.reduce_mean(-self.A_D_predictions) + self.lambda_B * self.B_loss

        self.B_D_predictions_ = self.B_d_net(self.real_A, reuse=True)
        self.B_d_loss_real = tf.reduce_mean(-self.B_D_predictions_)
        self.B_d_loss_fake = tf.reduce_mean(self.B_D_predictions)

        self.B_d_loss = self.B_d_loss_fake + self.B_d_loss_real
        self.B_g_loss = tf.reduce_mean(-self.B_D_predictions) + self.lambda_A * self.A_loss

        self.d_loss = self.A_d_loss + self.B_d_loss
        self.g_loss = self.A_g_loss + self.B_g_loss
        """
        self.translated_A_sum = tf.summary.image("translated_A", self.translated_A)
        self.translated_B_sum = tf.summary.image("translated_B", self.translated_B)
        self.recover_A_sum = tf.summary.image("recover_A", self.recover_A)
        self.recover_B_sum = tf.summary.image("recover_B", self.recover_B)
        """
        # define summary
        self.A_d_loss_sum = tf.summary.scalar("A_d_loss", self.A_d_loss)
        self.A_loss_sum = tf.summary.scalar("A_loss", self.A_loss)
        self.B_d_loss_sum = tf.summary.scalar("B_d_loss", self.B_d_loss)
        self.B_loss_sum = tf.summary.scalar("B_loss", self.B_loss)

        self.A_g_loss_sum = tf.summary.scalar("A_g_loss", self.A_g_loss)
        self.B_g_loss_sum = tf.summary.scalar("B_g_loss", self.B_g_loss)

        self.d_loss_sum = tf.summary.merge([self.A_d_loss_sum, self.B_d_loss_sum])
        self.g_loss_sum = tf.summary.merge([self.A_g_loss_sum, self.B_g_loss_sum, self.A_loss_sum, self.B_loss_sum])

        # define trainable variables
        t_vars = tf.trainable_variables()

        self.A_d_vars = [var for var in t_vars if 'A_d_' in var.name]
        self.B_d_vars = [var for var in t_vars if 'B_d_' in var.name]

        self.A_g_vars = [var for var in t_vars if 'A_g_' in var.name]
        self.B_g_vars = [var for var in t_vars if 'B_g_' in var.name]

        self.d_vars = self.A_d_vars + self.B_d_vars

        self.g_vars = self.A_g_vars + self.B_g_vars

    # def clip_trainable_vars(self, var_list):
    #     for var in var_list:
    #         self.sess.run(var.assign(tf.clip_by_value(var, -self.c, self.c)))

    def discriminator(self, image, y=None, prefix='A_', reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse is False

            h0 = lrelu(conv2d(image, self.df_dim, name=prefix + 'd_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(batch_norm(conv2d(h0, self.df_dim * 2, name=prefix + 'd_h1_conv'),
                                  self.is_training,
                                  name=prefix + 'd_bn1'))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 4, name=prefix + 'd_h2_conv'),
                                  self.is_training,
                                  name=prefix + 'd_bn2'))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(
                batch_norm(conv2d(h2, self.df_dim * 8, d_h=1, d_w=1, name=prefix + 'd_h3_conv'),
                           self.is_training,
                           name=prefix + 'd_bn3'))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, prefix + 'd_h3_lin')

            return h4

    def A_d_net(self, imgs, y=None, reuse=False):
        return self.discriminator(imgs, prefix='A_', reuse=reuse)

    def B_d_net(self, imgs, y=None, reuse=False):
        return self.discriminator(imgs, prefix='B_', reuse=reuse)

    def A_g_net(self, imgs, reuse=False):
        return self.fcn(imgs, prefix='A_g_', reuse=reuse)

    def B_g_net(self, imgs, reuse=False):
        return self.fcn(imgs, prefix='B_g_', reuse=reuse)

    def fcn(self, imgs, prefix=None, reuse=False):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse is False

            s = self.image_size
            s2, s4, s8, s16, s32, s64, s128 = \
                int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), \
                int(s / 64), int(s / 128)

            # imgs is (256 x 256 x input_c_dim)
            e1 = conv2d(imgs, self.fcn_filter_dim, name=prefix + 'e1_conv')
            # e1 is (128 x 128 x self.fcn_filter_dim)
            e2 = batch_norm(conv2d(lrelu(e1), self.fcn_filter_dim * 2, name=prefix + 'e2_conv'),
                            self.is_training,
                            name=prefix + 'bn_e2')
            # e2 is (64 x 64 x self.fcn_filter_dim*2)
            e3 = batch_norm(conv2d(lrelu(e2), self.fcn_filter_dim * 4, name=prefix + 'e3_conv'),
                            self.is_training,
                            name=prefix + 'bn_e3')
            # e3 is (32 x 32 x self.fcn_filter_dim*4)
            e4 = batch_norm(conv2d(lrelu(e3), self.fcn_filter_dim * 8, name=prefix + 'e4_conv'),
                            self.is_training,
                            name=prefix + 'bn_e4')
            # e4 is (16 x 16 x self.fcn_filter_dim*8)
            e5 = batch_norm(conv2d(lrelu(e4), self.fcn_filter_dim * 8, name=prefix + 'e5_conv'),
                            self.is_training,
                            name=prefix + 'bn_e5')
            # e5 is (8 x 8 x self.fcn_filter_dim*8)
            e6 = batch_norm(conv2d(lrelu(e5), self.fcn_filter_dim * 8, name=prefix + 'e6_conv'),
                            self.is_training,
                            name=prefix + 'bn_e6')
            # e6 is (4 x 4 x self.fcn_filter_dim*8)
            e7 = batch_norm(conv2d(lrelu(e6), self.fcn_filter_dim * 8, name=prefix + 'e7_conv'),
                            self.is_training,
                            name=prefix + 'bn_e7')
            # e7 is (2 x 2 x self.fcn_filter_dim*8)
            e8 = batch_norm(conv2d(lrelu(e7), self.fcn_filter_dim * 8, name=prefix + 'e8_conv'),
                            self.is_training,
                            name=prefix + 'bn_e8')
            # e8 is (1 x 1 x self.fcn_filter_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                                                     [self.batch_size, s128, s128, self.fcn_filter_dim * 8],
                                                     name=prefix + 'd1',
                                                     with_w=True)
            d1 = tf.nn.dropout(batch_norm(self.d1, self.is_training, name=prefix + 'bn_d1'),
                               0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.fcn_filter_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                                                     [self.batch_size, s64, s64, self.fcn_filter_dim * 8],
                                                     name=prefix + 'd2',
                                                     with_w=True)
            d2 = tf.nn.dropout(batch_norm(self.d2, self.is_training, name=prefix + 'bn_d2'),
                               0.5)

            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.fcn_filter_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                                                     [self.batch_size, s32, s32, self.fcn_filter_dim * 8],
                                                     name=prefix + 'd3',
                                                     with_w=True)
            d3 = tf.nn.dropout(batch_norm(self.d3, self.is_training, name=prefix + 'bn_d3'),
                               0.5)

            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.fcn_filter_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                                                     [self.batch_size, s16, s16, self.fcn_filter_dim * 8],
                                                     name=prefix + 'd4',
                                                     with_w=True)
            d4 = batch_norm(self.d4,
                            self.is_training,
                            name=prefix + 'bn_d4')

            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.fcn_filter_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                                                     [self.batch_size, s8, s8, self.fcn_filter_dim * 4],
                                                     name=prefix + 'd5',
                                                     with_w=True)
            d5 = batch_norm(self.d5,
                            self.is_training,
                            name=prefix + 'bn_d5')
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.fcn_filter_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                                                     [self.batch_size, s4, s4, self.fcn_filter_dim * 2],
                                                     name=prefix + 'd6',
                                                     with_w=True)
            d6 = batch_norm(self.d6,
                            self.is_training,
                            name=prefix + 'bn_d6')
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.fcn_filter_dim*2*2)

            self.d7, self.d7_w, self.d7_b = \
                deconv2d(tf.nn.relu(d6),
                         [self.batch_size, s2, s2, self.fcn_filter_dim],
                         name=prefix + 'd7',
                         with_w=True)
            d7 = batch_norm(self.d7,
                            self.is_training,
                            name=prefix + 'bn_d7')
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.fcn_filter_dim*1*2)

            if prefix == 'B_g_':
                self.d8, self.d8_w, self.d8_b = \
                    deconv2d(tf.nn.relu(d7),
                             [self.batch_size, s, s, self.input_channels_A],
                             name=prefix + 'd8',
                             with_w=True)
            elif prefix == 'A_g_':
                self.d8, self.d8_w, self.d8_b = \
                    deconv2d(tf.nn.relu(d7),
                             [self.batch_size, s, s, self.input_channels_B],
                             name=prefix + 'd8',
                             with_w=True)
            # d8 is (256 x 256 x output_c_dim)
            return tf.nn.tanh(self.d8)
