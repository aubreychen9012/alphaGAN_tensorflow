import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import scipy.ndimage

_eps = 1e-15

input_dim = 32
level = 3
z_dim = 10
n_l1 = 200
n_l2 = 400


def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


# same image size
def ResBlock(inputs, filter_in, filter_out, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='inputs')
        conv1 = Conv2d(input_layer, filter_in, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="conv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn1')
        conv2 = Conv2d(conv1, filter_out, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init, name="conv2")
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(input_layer, filter_out, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="conv3")
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn3')

        conv_out = conv2.outputs + conv3.outputs
    return conv_out


def ResBlockDown(inputs, filters, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='inputs')
        conv1 = Conv2d(input_layer, filters, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="conv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn1')
        conv2 = Conv2d(conv1, filters * 2, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init, name="conv2")
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(input_layer, filters * 2, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="conv3")
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn3')

        conv_out = conv2.outputs + conv3.outputs
    return conv_out


# image size *2
def ResBlockUp(inputs, input_size, batch_size, filters, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='inputs')
        conv1 = DeConv2d(input_layer, filters, (3, 3), (input_size * 2, input_size * 2), (2, 2),
                         batch_size=batch_size, act=None, padding='SAME',
                         W_init=w_init, b_init=b_init, name="deconv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn1')
        conv2 = DeConv2d(conv1, filters / 2, (3, 3), (input_size * 2, input_size * 2), (1, 1), act=None, padding='SAME',
                         batch_size=batch_size, W_init=w_init, b_init=b_init, name="deconv2")
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn2')

        conv3 = DeConv2d(input_layer, filters / 2, (3, 3), (input_size * 2, input_size * 2), (2, 2), act=None,
                         padding='SAME',
                         batch_size=batch_size, W_init=w_init, b_init=b_init, name="conv3")
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn3')

        conv_out = conv2.outputs + conv3.outputs
    return conv_out

def encoder(x, reuse=False, is_train=True):
    """
    Encode part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    image_size = input_dim
    s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)
    gf_dim = 16  # Dimension of gen filters in first conv layer. [64]
    ft_size = 3
    # c_dim = FLAGS.c_dim  # n_color 3
    # batch_size = 64  # 64
    with tf.variable_scope("ae_encoder", reuse=reuse):
        # x,y,z,_ = tf.shape(input_images)
        tl.layers.set_name_reuse(reuse)

        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.01)

        inputs = InputLayer(x, name='e_inputs')
        conv1 = Conv2d(inputs, gf_dim, (ft_size, ft_size), act=lambda x: tl.act.lrelu(x, 0.2), padding='SAME',
                       W_init=w_init, b_init=b_init,
                       name="e_conv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                               gamma_init=gamma_init, name='e_bn1')
        # image_size * image_size
        res1 = ResBlockDown(conv1.outputs, gf_dim, "res1", reuse, is_train)
        # res1 = tf.layers.conv2d(inputs=res1, filters = gf_dim*2, kernel_size = (ft_size,ft_size), strides=(2,2),
        #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
        #                        trainable=True, name='res1_downsample')

        # s2*s2
        res2 = ResBlockDown(res1, gf_dim * 2, "res2", reuse, is_train)
        # res2 = tf.layers.conv2d(inputs=res2, filters = gf_dim*4, kernel_size = (ft_size,ft_size), strides=(2,2),
        #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
        #                        trainable=True, name='res2_downsample')

        # s4*s4
        res3 = ResBlockDown(res2, gf_dim * 4, "res3", reuse, is_train)
        # res3 = tf.layers.conv2d(inputs=res3, filters = gf_dim*8, kernel_size = (ft_size,ft_size), strides=(2,2),
        #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
        #                        trainable=True, name='res3_downsample')

        # s8*s8
        res4 = ResBlockDown(res3, gf_dim * 8, "res4", reuse, is_train)
        # res4 = tf.layers.conv2d(inputs=res4, filters = gf_dim*16, kernel_size = (ft_size,ft_size), strides=(2,2),
        #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
        #                        trainable=True, name='res4_downsample')

        # s16*s16
        h_flat = tf.reshape(res4, shape=[-1, s16 * s16 * gf_dim * 16])
        h_flat = InputLayer(h_flat, name='e_reshape')
        net_h = DenseLayer(h_flat, n_units=z_dim, act=tf.identity, name="e_dense_mean")
    return net_h.outputs

def decoder(x, reuse=False, is_train=True):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    image_size = input_dim
    s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)
    gf_dim = 16  # Dimension of gen filters in first conv layer. [64]
    c_dim = 1  # n_color 3
    ft_size = 3
    batch_size = 64  # 64
    with tf.variable_scope("ae_decoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        # gamma_init = tf.random_normal_initializer(1., 0.02)
        # weights_gener = dict()
        inputs = InputLayer(x, name='g_inputs')

        # s16*s16
        z_develop = DenseLayer(inputs, s16 * s16 * gf_dim * 16, act=lambda x: tl.act.lrelu(x, 0.2),
                               name='g_dense_z')
        z_develop = tf.reshape(z_develop.outputs, [-1, s16, s16, gf_dim * 16])
        z_develop = InputLayer(z_develop, name='g_reshape')
        conv1 = Conv2d(z_develop, gf_dim * 8, (ft_size, ft_size), act=lambda x: tl.act.lrelu(x, 0.2),
                       padding='SAME',
                       W_init=w_init, b_init=b_init, name="g_conv1")

        # s16*s16
        res1 = ResBlockUp(conv1.outputs, s16, batch_size, gf_dim * 8, "gres1", reuse, is_train)
        # res1 = tf.layers.conv2d_transpose(inputs=res1, filters = gf_dim*4, kernel_size = (ft_size, ft_size), strides = (2,2),
        #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
        #                                 kernel_initializer=w_init, trainable=True, name='res1_upsample')

        # s8*s8
        res2 = ResBlockUp(res1, s8, batch_size, gf_dim * 4, "gres2", reuse, is_train)
        # res2 = tf.layers.conv2d_transpose(inputs=res2, filters=gf_dim*2, kernel_size=(ft_size, ft_size), strides=(2, 2),
        #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
        #                                  kernel_initializer=w_init, trainable=True, name='res2_upsample')

        # s4*s4
        res3 = ResBlockUp(res2, s4, batch_size, gf_dim * 2, "gres3", reuse, is_train)
        # res3 = tf.layers.conv2d_transpose(inputs=res3, filters=gf_dim, kernel_size=(ft_size, ft_size), strides=(2, 2),
        #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
        #                                  kernel_initializer=w_init, trainable=True, name='res3_upsample')

        # s2*s2
        res4 = ResBlockUp(res3, s2, batch_size, gf_dim, "gres4", reuse, is_train)
        # res4 = tf.layers.conv2d_transpose(inputs=res4, filters=8, kernel_size=(ft_size, ft_size), strides=(2, 2),
        #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
        #                                  kernel_initializer=w_init, trainable=True, name='res4_upsample')
        # image_size*image_size
        res_inputs = InputLayer(res4, name='res_inputs')
        conv2 = Conv2d(res_inputs, c_dim, (ft_size, ft_size), act=tf.nn.sigmoid, padding='SAME', W_init=w_init,
                       b_init=b_init,
                       name="g_conv2")
        conv2_std = Conv2d(res_inputs, c_dim, (ft_size, ft_size), act=None, padding='SAME', W_init=w_init,
                           b_init=b_init,
                           name="g_conv2_std")

        # deconv1 = DeConv2d(res_inputs, c_dim, (3, 3), out_size=(image_size, image_size), strides=(1, 1),
        #                   padding="SAME", act=None, batch_size=batch_size, W_init=w_init, b_init=b_init,
        #                   name="g_mu_output")
        # deconv1 = DeConv2d(res_inputs, c_dim, (3, 3), out_size=(image_size, image_size), strides=(1, 1),
        #                   padding="SAME", act=None, batch_size=batch_size, W_init=w_init, b_init=b_init,
        #                   name="g_std_output")
        #logits = conv1.outputs
    return conv2.outputs, conv2_std.outputs

def discriminator(x, reuse=False):
    """
    Discriminator that is used to match the posterior distribution with a given prior distribution.
    :param x: tensor of shape [batch_size, z_dim]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    with tf.variable_scope("discriminator", reuse=reuse):
        w_init = tf.random_normal_initializer(stddev=0.01)
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(x, name='dc/in')
        net_h0 = DenseLayer(net_in, n_units=n_l1,
                            W_init=w_init,
                            act=lambda x: tl.act.lrelu(x, 0.2), name='dc/h0/lin')
        net_h1 = DenseLayer(net_h0, n_units=n_l2,
                            W_init=w_init,
                            act=lambda x: tl.act.lrelu(x, 0.2), name='dc/h1/lin')
        net_h2 = DenseLayer(net_h1, n_units=1,
                            W_init=w_init,
                            act=tf.identity, name='dc/h2/lin')
        # net_h1 = DenseLayer(net_h1, n_units=n_l2, W_init=w_init,
        #                        act=tf.nn.elu, name='dc/h2/lin')
        # net_h2 = DenseLayer(net_h1, n_units=1, W_init=w_init,
        #                    act=tf.nn.relu, name='dc/h2/lin')
        logits = net_h2.outputs
        net_h2.outputs = tf.nn.sigmoid(net_h2.outputs)
        return net_h2.outputs, logits

def discriminate_decoder(input, reuse, istrain):
    df_dim = 16
    ft_size = 3
    image_size = input_dim
    with tf.variable_scope("discriminate_decoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.random_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)
        inputs = InputLayer(input, name='dcd/inputs')
        conv1 = Conv2d(inputs, df_dim, (ft_size, ft_size), W_init=w_init, b_init=b_init,
                       act=lambda x: tl.act.lrelu(x, 0.2), name='conv1')
        conv2 = Conv2d(conv1, df_dim*2, (ft_size, ft_size), W_init=w_init, b_init=b_init,
                       act=lambda x: tl.act.lrelu(x, 0.2), name='conv2')
        conv3 = Conv2d(conv2, df_dim*4, (ft_size, ft_size), W_init=w_init, b_init=b_init,
                       act=lambda x: tl.act.lrelu(x, 0.2), name='conv3')
        # conv1 = MeanPool2d(conv1, (2, 2), name='avgpool1')  # image_size/2
        #block1 = ResBlockDown(conv1.outputs, df_dim, "res1", reuse, istrain)
        # block1 = MeanPool2d(block1, (2, 2), name='avgpool_res1')  # image_size/4
        #block2 = ResBlockDown(block1, df_dim*2, scope_name="res2", reuse=reuse, phase_train=istrain)
        # block2 = MeanPool2d(block2, (2, 2), name='avgpool_res2')  # image_size/8
        #block3 = ResBlockDown(block2, df_dim*4, scope_name="res3", reuse=reuse, phase_train=istrain)

        #block3_flat = tf.reshape(block3, shape=[-1,
        #                                  image_size / 8 * image_size / 8 * df_dim *8])
        block3_flat = tf.reshape(conv3.outputs, shape=[-1, image_size*image_size*df_dim*4])
        block3_inputs = InputLayer(block3_flat, name='res3_inputs')
        label_logits = DenseLayer(block3_inputs, n_units=1, act=tf.identity, name='d_dense')
        label = tf.nn.sigmoid(label_logits.outputs)
    return label, label_logits.outputs




# class Networks():
#     def __init__(self, input,batch_size, z_dim, FLAGS, reuse=False, pad='SAME'):
#         self.input = input
#         self.image_size = tf.shape(input)[-2]
#         self.channel_num = tf.shape(input)[-1]
#         self.batch_size = batch_size
#         self.z_dim = z_dim
#         self.w_init = tf.truncated_normal_initializer(stddev=0.02)
#         self.b_init = tf.constant_initializer(value=0.0)
#         self.gamma_init = tf.random_normal_initializer(1., 0.02)
#         self.pad = pad
#
#     def encode(self, reuse, istrain):
#         with tf.variable_scope("encoder",reuse=reuse):
#             inputs = InputLayer(self.input, name='inputs')
#             conv1 = Conv2d(inputs, 128, (5,5), W_init=self.w_init, b_init=self.b_init, padding=self.pad, act=tf.nn.relu, name='conv1')
#             conv1 = MeanPool2d(conv1, (2,2), name='avgpool1') # image_size/2
#             block1 = ResBlock(conv1, w_init=self.w_init, b_init=self.b_init, gamma_init=self.gamma_init,
#                               is_train=istrain, scope_name="block1")
#             block1 = MeanPool2d(block1, (2,2), name='avgpool_res1') #image_size/4
#             block2 = ResBlock(block1, w_init=self.w_init, b_init=self.b_init, gamma_init=self.gamma_init,
#                               is_train=istrain, scope_name="block2")
#             block2 = MeanPool2d(block2, (2,2), name='avgpool_res2') # image_size/8
#             block3 = ResBlock(block2, w_init=self.w_init, b_init=self.b_init, gamma_init=self.gamma_init,
#                               is_train=istrain, scope_name="block3")
#             block3_flat = tf.reshape(block3, [self.batch_size, self.image_size/8*self.image_size/8*tf.shape(block3)[-1]])
#             z = DenseLayer(block3_flat, self.z_dim, act=None, name='z_dense')
#             self.latent_vars = z
#         return z
#
#     def generate(self, reuse,istrain):
#         with tf.variable_scope("generator", reuse=reuse):
#             inputs = InputLayer(self.latent_vars, name='inputs')
#             z = DenseLayer(inputs, self.image_size/8*self.image_size/8*128,
#                          act=tf.nn.relu, name="z_dense",)
#             z = tf.reshape(z, [self.batch_size, self.image_size/8, self.image_size/8, 128])
#             block1 = ResBlock(z, w_init=self.w_init, b_init=self.b_init, gamma_init=self.gamma_init,
#                               is_train=istrain, scope_name="block1")
#             block1 = tf.image.resize_images(block1, self.image_size/4,self.image_size/4)
#             block2 = ResBlock(block1, w_init=self.w_init, b_init=self.b_init, gamma_init=self.gamma_init,
#                               is_train=istrain, scope_name="block2")
#             block2 = tf.image.resize_images(block2, self.image_size/2, self.image_size/2)
#             block3 = ResBlock(block1, w_init=self.w_init, b_init=self.b_init, gamma_init=self.gamma_init,
#                               is_train=istrain, scope_name="block3")
#             block3 = tf.image.resize_images(block3, self.image_size, self.image_size)
#             block4 = ResBlock(block1, w_init=self.w_init, b_init=self.b_init, gamma_init=self.gamma_init,
#                               is_train=istrain, scope_name="block4")
#             conv4 = Conv2d(block4, self.channel_num, (5,5), W_init=self.w_init, b_init=self.b_init, act=None,name='conv4')
#         return conv4
#
#     ## discriminator for real images, reconstructed images and generated images
#     def discriminate(self, reuse, istrain):
#         with tf.variable_scope("discriminator", reuse=reuse):
#             inputs = InputLayer(self.input, name='inputs')
#             conv1 = Conv2d(inputs, 128, (5, 5), W_init=self.w_init, b_init=self.b_init, act=tf.nn.relu, name='conv1')
#             conv1 = MeanPool2d(conv1, (2, 2), name='avgpool1')  # image_size/2
#             block1 = ResBlock(conv1, w_init=self.w_init, b_init=self.b_init, gamma_init=self.gamma_init,
#                               is_train=istrain, scope_name="block1")
#             block1 = MeanPool2d(block1, (2, 2), name='avgpool_res1')  # image_size/4
#             block2 = ResBlock(block1, w_init=self.w_init, b_init=self.b_init, gamma_init=self.gamma_init,
#                               is_train=istrain, scope_name="block2")
#             block2 = MeanPool2d(block2, (2, 2), name='avgpool_res2')  # image_size/8
#             block3 = ResBlock(block2, w_init=self.w_init, b_init=self.b_init, gamma_init=self.gamma_init,
#                               is_train=istrain, scope_name="block3")
#             block3_flat = tf.reshape(block3, [self.batch_size,
#                                               self.image_size / 8 * self.image_size / 8 * tf.shape(block3)[-1]])
#             label = DenseLayer(block3_flat, 1, act=tf.nn.sigmoid, name='d_dense')
#         return label
#
#     ## classifier to code and generated code
#     def classify(self, reuse, istrain):
#         with tf.variable_scope("classifier", reuse=reuse):
#             inputs = InputLayer(self.latent_vars, name='inputs')
#             dense1 = DenseLayer(inputs, 700, act=lrelu, name='dense1')
#             dense2 = DenseLayer(dense1, 700, act=lrelu, name='dense2')
#             dense3 = DenseLayer(dense2, 1, act=tf.nn.sigmoid, name='dense3')
#         return dense3