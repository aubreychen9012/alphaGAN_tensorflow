import os, sys, pprint, time
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SGE_GPU'][-1]
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from models import encoder, decoder, discriminator, discriminate_decoder
from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imresize

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 2e-5, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 32, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 32, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("input_dim", 32, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 10, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS

lambda_=1.5
level=3
results_path = './Results/'

mnist = input_data.read_data_sets('./Data', one_hot=True)

def resize(img_3d,scale):
    img=img_3d
    res = []
    for i in img:
        i_ = imresize(i,scale, interp='nearest')
        i_re = (i_/255.0)*(i.max()-i.min())+i.min()
        res.append(i_re)
    return np.asarray(res)

def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    results_path = './Results/'
    folder_name = "/{0}_{1}_{2}_alphaGAN". \
        format(FLAGS.image_size, FLAGS.z_dim, lambda_)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


def train(train_model=True,load =0, comment=None, model_name=None, modelstep=0):
    save_dir="checkpoint"
    log_dir = "logs"
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir(log_dir)
    # task = "model_dataset" + str(FLAGS.dataset)+"_image_size"+str(FLAGS.image_size)\
    #        +"_z_dim"+str(FLAGS.z_dim)+"_learning_rate_"+str(FLAGS.learning_rate)+\
    #        "_epoch_"+str(FLAGS.epoch)+"_batchsize_"+str(FLAGS.batch_size)
    #
    # tl.files.exists_or_mkdir("samples/{}".format(task))
    with tf.device("/gpu:0"):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device("/gpu:0"):
            #z = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.z_dim], name='encoded_z')
            real_distribution = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.z_dim], name='Real_distribution')
            real_images = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim],name='real_images')
            #===============================================#

            encoder_output = encoder(real_images, reuse=False, is_train=True)
            encoder_output_test = encoder(real_images, reuse=True, is_train=False)
            d_fake, d_fake_logits = discriminator(encoder_output, reuse=False)
            d_real, d_real_logits = discriminator(real_distribution, reuse=True)

            d_fake_test, d_fake_logits_test = discriminator(encoder_output_test, reuse=True)
            d_real_test, d_real_logits_test = discriminator(real_distribution, reuse=True)

            decoder_output, std = decoder(encoder_output, reuse=False, is_train=True)
            # encoder_output_z = encoder(decoder_output, reuse=True, is_train=False)
            decoder_output_test, std_ = decoder(encoder_output, reuse=True, is_train=False)
            # encoder_output_z_test = encoder(decoder_output_test, reuse=True, is_train=False)
            decoder_z_output,_ = decoder(real_distribution, reuse=True, is_train=False)

            d_fake_decoder, d_fake_decoder_logits = discriminate_decoder(decoder_output, reuse=False, istrain=True)
            d_real_decoder, d_real_decoder_logits = discriminate_decoder(real_images, reuse=True, istrain=False)

            d_fake_decoder_test, d_fake_decoder_logits_test = discriminate_decoder(decoder_output_test, reuse=True,
                                                                                   istrain=False)
            d_real_decoder_test, d_real_decoder_logits_test = discriminate_decoder(real_images, reuse=True,
                                                                                   istrain=False)
            d_sample_decoder, d_sample_decoder_logits = discriminate_decoder(decoder_z_output, reuse=True,
                                                                             istrain=False)

            summed = tf.reduce_sum(tf.square(decoder_output - real_images), [1, 2, 3])
            # sqrt_summed = summed
            sqrt_summed = tf.sqrt(summed + 1e-8)
            autoencoder_loss = tf.reduce_mean(sqrt_summed)
            summed_test = tf.reduce_sum(tf.square(decoder_output_test - real_images), [1, 2, 3])
            # sqrt_summed_test = summed_test
            sqrt_summed_test = tf.sqrt(summed_test + 1e-8)
            autoencoder_loss_test = tf.reduce_mean(sqrt_summed_test)

            # discriminator loss
            tf_randn_real = tf.random_uniform(tf.shape(d_real), 0.8, 1.1)
            tf_randn_fake = tf.random_uniform(tf.shape(d_real), 0.0, 0.1)

            dc_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_randn_real, logits=d_real_logits))
            dc_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_randn_fake, logits=d_fake_logits))
            dc_loss = dc_loss_fake + dc_loss_real


            dc_loss_real_test = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_randn_real, logits=d_real_logits_test))
            dc_loss_fake_test = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake_logits_test))
            dc_loss_test = dc_loss_fake_test + dc_loss_real_test


            # Generator loss
            generator_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake_logits))
            generator_loss_test = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake_test), logits=d_fake_logits_test))

            # decoder's discriminator loss
            tf_randn_real = tf.random_uniform(tf.shape(d_real_decoder), 0.8, 1.1)
            tf_randn_fake = tf.random_uniform(tf.shape(d_fake_decoder), 0.0, 0.1)

            dc_decoder_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_randn_real, logits=d_real_decoder_logits))
            dc_decoder_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake_decoder_logits))
            dc_decoder_loss_sample = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake),
                                                        logits=d_sample_decoder_logits))
            dc_decoder_loss = dc_decoder_loss_fake + dc_decoder_loss_real + dc_decoder_loss_sample
            # dc_decoder_loss = dc_decoder_loss_fake + dc_decoder_loss_real + dc_loss_decoder_sample
            #dc_decoder_loss = tf.reduce_mean(-tf.log(d_real_decoder+1e-8) - tf.log(1 - d_fake_decoder+1e-8)
            #                                 - tf.log(1 - d_sample_decoder+1e-8))

            dc_decoder_loss_real_test = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_randn_real, logits=d_real_decoder_logits_test))
            dc_decoder_loss_fake_test = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake_decoder_logits_test))
            dc_decoder_loss_sample_test = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake),
                                                        logits=d_sample_decoder_logits))
            # dc_loss_decoder_test = dc_loss_decoder_fake_test + dc_loss_decoder_real_test + dc_loss_decoder_sample_test
            #dc_loss_decoder_test = tf.reduce_mean(-tf.log(d_real_decoder_test+1e-8) - tf.log(1 - d_fake_decoder_test+1e-8)
            #                                      - tf.log(1 - d_sample_decoder+1e-8))
            dc_decoder_loss_test = dc_decoder_loss_fake_test + dc_decoder_loss_real_test + dc_decoder_loss_sample

            # decoder's generator loss
            generator_decoder_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake_decoder), logits=d_fake_decoder_logits))
            generator_decoder_sample_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_sample_decoder),
                                                        logits=d_sample_decoder_logits))
            generator_decoder_loss_test = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake_decoder_test), logits=d_fake_decoder_logits_test))

            #dc_decoder_loss = tf.reduce_mean(d_real_decoder_logits - d_fake_decoder_logits)
            #dc_loss_decoder_test = tf.reduce_mean(d_real_decoder_logits_test - d_fake_decoder_logits_test)

            E_vars = tl.layers.get_variables_with_name('encoder', True, True)
            G_vars = tl.layers.get_variables_with_name('decoder', True, True)
            EG_vars = tl.layers.get_variables_with_name('ae', True, True)
            D_vars = tl.layers.get_variables_with_name('discriminator', True, True)
            C_vars = tl.layers.get_variables_with_name('discriminate_decoder', True, True)

            lr_v = FLAGS.learning_rate
            dc = tf.log(tf.divide(d_fake_logits, 1-d_fake_logits+1e-8)+1e-8)
            dcd = tf.log(tf.divide(d_fake_decoder_logits, 1-d_fake_decoder_logits+1e-8)+1e-8)

            #generator_loss = lambda_*autoencoder_loss - tf.log(d_fake_logits+1e-8) + tf.log(1-d_fake_logits+1e-8)
            # for E vars
            def generator_loss_func(x):
                return tf.reduce_mean(-tf.log(x+1e-8) + tf.log(1.-x+1e-8))

            #encoder_loss = lambda_*autoencoder_loss + generator_loss_func(d_fake)
            encoder_loss = lambda_ * autoencoder_loss + generator_loss
            #encoder_loss_test = lambda_ * autoencoder_loss_test + generator_loss_func(d_fake_test)
            encoder_loss_test = lambda_ * autoencoder_loss + generator_loss_test

            # for G vars, generator loss in the paper
            #decoder_loss = lambda_*autoencoder_loss + generator_loss_func(d_fake_decoder) + \
            #                    generator_loss_func(d_sample_decoder)
            decoder_loss = lambda_ * autoencoder_loss + generator_decoder_loss + \
                           generator_decoder_sample_loss
            #decoder_loss_test = lambda_ * autoencoder_loss_test + generator_loss_func(d_fake_decoder_test) + \
            #                    generator_loss_func(d_sample_decoder)
            decoder_loss_test = lambda_ * autoencoder_loss_test + generator_decoder_loss_test + \
                                generator_decoder_sample_loss

                # tf.log(d_fake_test + 1e-8)
                #                                                                  + tf.log(1 - d_fake_test + 1e-8))
            # for C vars, discriminator for the decoder
            # dc_decoder_loss
            # D vars
            # dc_loss

        with tf.device("/gpu:0"):
            train_op_e = tf.train.AdamOptimizer(lr_v, beta1=FLAGS.beta1).minimize(encoder_loss,
                                                                                var_list=E_vars)
            train_op_g = tf.train.AdamOptimizer(lr_v, beta1=FLAGS.beta1).minimize(decoder_loss,
                                                                                var_list=G_vars)
            train_op_d = tf.train.AdamOptimizer(lr_v, beta1=FLAGS.beta1).minimize(dc_loss,
                                                                                var_list=D_vars)
            train_op_dc = tf.train.AdamOptimizer(lr_v, beta1=FLAGS.beta1).minimize(dc_decoder_loss,
                                                                                var_list=C_vars)
            tl.layers.initialize_global_variables(sess)

        tensorboard_path, saved_model_path, log_path = form_results()
        input_images = tf.reshape(real_images, [-1, FLAGS.input_dim, FLAGS.input_dim, 1])
        generated_images = tf.reshape(decoder_output, [-1, FLAGS.input_dim, FLAGS.input_dim, 1])

        writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=tf.get_default_graph())
        tf.summary.scalar("autoencoder_loss", autoencoder_loss)
        tf.summary.scalar("discriminator_loss", dc_loss)
        tf.summary.scalar("generator_loss", generator_loss)
        tf.summary.scalar("discriminate_decoder_loss", dc_decoder_loss)
        tf.summary.scalar("discriminate_generator_loss", generator_decoder_loss)
        tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
        tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
        tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
        tf.summary.histogram(name='Real Distribution', values=real_distribution)
        summary_op=tf.summary.merge_all()
        saver=tf.train.Saver()

    if not train_model:
        # Get the latest results folder
        all_results = os.listdir(results_path)
        all_results.sort()
        #saver.restore(sess,
        #              save_path=tf.train.latest_checkpoint(results_path + '/' + all_results[-1] + '/Saved_models/'))
        return all_results

        # tl.layers.initialize_global_variables(sess)
        # ## load existing model if possible
        # tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/{}.npz'.format(task), network=alphagan)
    
    #X_train, y_train = datasets.create_datasets(retrain=0, task="gan_" + str(FLAGS.z_dim) + "_" + str(FLAGS.input_dim))
    #x,y = brats.create_datasets(retrain=0, task="brats_aae_wgan_" + str(z_dim) + "_" + str(input_dim))
    # bp()
    #X_train_lowres = lowres_level(X_train, level).astype("float32")
    #y_train_lowres = lowres_level(y_train, level).astype("float32")

    step=0
    with open(log_path + '/log.txt', 'a') as log:
        log.write("Comment: {}\n".format(comment))
        log.write("\n")
        log.write("input_dim: {}\n".format(FLAGS.input_dim))
        log.write("z_dim: {}\n".format(FLAGS.z_dim))
        log.write("batch_size: {}\n".format(FLAGS.batch_size))
        log.write("learning_rate: {}\n".format(FLAGS.learning_rate))
        log.write("\n")

    for i in range(FLAGS.epoch):
        n_batches = int(mnist.train.num_examples / FLAGS.batch_size)
        # b = 0
        for b in range(1, n_batches + 1):
            batch_x, _ = mnist.train.next_batch(FLAGS.batch_size)
            batch_x = batch_x.reshape(FLAGS.batch_size, 28, 28)
            batch_x = resize(batch_x, 32.0 / 28.0)
            batch_x = batch_x[:,:,:,np.newaxis]
            images= batch_x
            z_real_dist=np.random.normal(0, 1, (FLAGS.batch_size, FLAGS.z_dim)) * 1.
            sess.run(train_op_e, {real_images:images,real_distribution:z_real_dist})
            sess.run(train_op_g, {real_images:images, real_distribution:z_real_dist})
            sess.run(train_op_d, {real_images:images, real_distribution:z_real_dist})
            #for _ in range(2):
            sess.run(train_op_dc, {real_images:images, real_distribution:z_real_dist})
            if b%20==0:
                e_loss, d_loss, dc_loss, g_loss, dcd_loss, gd_loss, ae_loss, summary = sess.run(
                    [encoder_loss_test, decoder_loss_test, dc_loss_test, generator_loss_test, dc_decoder_loss_test,
                     generator_decoder_loss_test,autoencoder_loss,summary_op],
                    feed_dict={real_images: images,
                               real_distribution: z_real_dist})
                df_decoder, df_decoder_test = sess.run(
                    [d_fake_decoder, d_fake_decoder_test],
                    feed_dict={real_images: images,
                               real_distribution: z_real_dist})
                writer.add_summary(summary, global_step=step)
                print("Epoch: {}, iteration: {}".format(i, b))
                print("Encoder Loss: {}".format(e_loss))
                print("Decoder Loss: {}".format(d_loss))
                print("Discriminator Loss: {}".format(dc_loss))
                print("Generator Loss: {}".format(g_loss))
                print("Discriminate decoder Loss: {}".format(dcd_loss))
                print("Discriminate generator Loss: {}".format(gd_loss))
                print("Autoencoder LOss:{}".format(ae_loss))
                print("df_decoder {}, df_decoder_test".format(df_decoder, df_decoder_test))
                with open('./logs/log.txt', 'a') as log:
                    log.write("Epoch: {}, iteration: {}\n".format(i, b))
                    log.write("Encoder Loss: {}\n".format(e_loss))
                    log.write("Decoder Loss: {}\n".format(d_loss))
                    log.write("Generator Loss:{}\n".format(g_loss))
                    log.write("Discriminator Loss: {}\n".format(dc_loss))
                    log.write("Discriminate decoder Loss:{}\n".format(dcd_loss))
                    log.write("Discriminate Generator Loss: {}\n".format(gd_loss))
                saver.save(sess, save_path=saved_model_path, global_step=step)
            #i+=1
            step+=1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=int, default=0, help='retrain model')
    parser.add_argument('--model_name', type=str, default='None', help='model to retrain on')
    parser.add_argument('--step', type=str, default='None', help='model to retrain on')
    parser.add_argument('--comment', type=str, default='None', help='model comment')
    args = parser.parse_args()
    train(train_model=True,load =args.load, comment=args.comment, model_name=args.model_name, modelstep=args.step)